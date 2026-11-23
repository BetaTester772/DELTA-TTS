import AVFoundation
import CoreAudio
import Foundation
import Observation

/// A microphone that can be chosen for recording.
struct InputDevice: Identifiable, Hashable {
    var id: String  // AVCaptureDevice.uniqueID (the CoreAudio device UID)
    var name: String
}

/// The audio devices of the Mac: the connected microphones and the system default input and
/// output, kept current through CoreAudio listeners (devices plugged in or out, defaults changed
/// in System Settings). Set DELTA_TTS_DEMO_DEBUG=1 to log them.
@MainActor
@Observable
final class AudioDevices {
    private(set) var inputs: [InputDevice] = []
    private(set) var defaultInputName: String?
    private(set) var defaultOutputName: String?
    private(set) var defaultOutputID = AudioDeviceID(0)

    /// Called when the system default output device changes.
    @ObservationIgnored var onDefaultOutputChange: (() -> Void)?

    private static let debug = ProcessInfo.processInfo.environment["DELTA_TTS_DEMO_DEBUG"] != nil

    init() {
        refresh()
        for selector in [kAudioHardwarePropertyDevices, kAudioHardwarePropertyDefaultInputDevice,
                         kAudioHardwarePropertyDefaultOutputDevice] {
            var address = Self.address(selector)
            AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, .main) { [weak self] _, _ in
                MainActor.assumeIsolated { self?.refresh() }
            }
        }
    }

    /// The microphone with this unique ID, if it is connected.
    func captureDevice(uniqueID: String) -> AVCaptureDevice? {
        AVCaptureDevice(uniqueID: uniqueID)
    }

    func refresh() {
        let discovery = AVCaptureDevice.DiscoverySession(deviceTypes: [.microphone, .external], mediaType: .audio,
                                                         position: .unspecified)
        inputs = discovery.devices.map { InputDevice(id: $0.uniqueID, name: $0.localizedName) }
        defaultInputName = AVCaptureDevice.default(for: .audio)?.localizedName
        let outputID = Self.defaultDevice(output: true)
        defaultOutputName = Self.name(of: outputID)
        if outputID != defaultOutputID {
            let changed = defaultOutputID != 0
            defaultOutputID = outputID
            if changed { onDefaultOutputChange?() }
        }
        if Self.debug {
            print("[audio] inputs: \(inputs.map { "\($0.name) <\($0.id)>" })")
            print("[audio] default input: \(defaultInputName ?? "none"); default output: \(defaultOutputName ?? "none") (#\(outputID))")
        }
    }

    // -----------------------------
    // CoreAudio
    // -----------------------------
    /// The system default output (or input) device.
    nonisolated static func defaultDevice(output: Bool) -> AudioDeviceID {
        var address = address(output ? kAudioHardwarePropertyDefaultOutputDevice : kAudioHardwarePropertyDefaultInputDevice)
        var device = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &device)
        return status == noErr ? device : 0
    }

    nonisolated static func name(of device: AudioDeviceID) -> String? {
        guard device != 0 else { return nil }
        var address = address(kAudioObjectPropertyName)
        var name: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let status = withUnsafeMutablePointer(to: &name) {
            AudioObjectGetPropertyData(device, &address, 0, nil, &size, $0)
        }
        guard status == noErr, let name else { return nil }
        return name.takeRetainedValue() as String
    }

    private nonisolated static func address(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                   mElement: kAudioObjectPropertyElementMain)
    }
}
