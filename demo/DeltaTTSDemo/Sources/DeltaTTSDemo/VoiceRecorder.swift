import AVFoundation
import CoreMedia
import Foundation
import Observation

/// Whether the app may use the microphone.
enum MicrophoneAccess: Equatable {
    case undetermined
    case granted
    case denied
    case needsAppBundle  // no NSMicrophoneUsageDescription: started with `swift run`, not as DeltaTTSDemo.app

    static var current: MicrophoneAccess {
        guard Bundle.main.object(forInfoDictionaryKey: "NSMicrophoneUsageDescription") != nil else { return .needsAppBundle }
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return .granted
        case .denied, .restricted: return .denied
        default: return .undetermined
        }
    }
}

/// A finished recording.
struct Take: Equatable {
    var url: URL
    var duration: TimeInterval
    var peakDecibels: Double  // loudest 20 ms frame, dBFS
}

/// Records the visitor's voice: a countdown, then mono 16-bit WAV from the given microphone
/// until Stop or the time limit, then leading and trailing silence is cut. It records through
/// AVCaptureSession, which, unlike AVAudioRecorder, can record from a chosen device.
///
/// Recordings live in a private temporary directory (mode 0700) that `deleteAll` empties and
/// `removeDirectory` removes; directories of earlier runs that did not quit cleanly are removed
/// at startup.
@MainActor
@Observable
final class VoiceRecorder {
    enum State: Equatable {
        case idle
        case countdown(Int)
        case recording
    }

    private(set) var state = State.idle
    private(set) var take: Take?
    private(set) var level = 0.0  // input level, 0...1
    private(set) var elapsed: TimeInterval = 0
    private(set) var maxSeconds: TimeInterval = 12
    private(set) var access = MicrophoneAccess.current
    private(set) var failure: String?  // why the last recording failed

    let directory: URL

    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var output: AVCaptureAudioFileOutput?

    nonisolated static let directoryPrefix = "DeltaTTSDemo-recordings-"

    init() {
        Self.removeStaleDirectories()
        directory = FileManager.default.temporaryDirectory
            .appending(path: "\(Self.directoryPrefix)\(getpid())-\(UUID().uuidString.prefix(8))", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                 attributes: [.posixPermissions: 0o700])
    }

    var isCapturing: Bool { state != .idle }

    /// Count down, then record from `device` until `stop` or `maxSeconds`.
    func start(device: AVCaptureDevice, countdown: Int, maxSeconds: TimeInterval, sampleRate: Double, trimSilence: Bool) {
        guard !isCapturing else { return }
        task = Task {
            await run(device: device, countdown: countdown, maxSeconds: maxSeconds, sampleRate: sampleRate,
                      trimSilence: trimSilence)
        }
    }

    /// Cancel the countdown, or end the recording and keep it.
    func stop() {
        switch state {
        case .countdown: task?.cancel()
        case .recording: output?.stopRecording()
        case .idle: break
        }
    }

    func refreshAccess() {
        access = MicrophoneAccess.current
    }

    /// Cancel any recording and delete every recorded file.
    func deleteAll() {
        task?.cancel()
        output?.stopRecording()
        take = nil
        failure = nil
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        for file in files { try? FileManager.default.removeItem(at: file) }
    }

    /// Delete the recordings and their directory (when the app quits).
    func removeDirectory() {
        deleteAll()
        try? FileManager.default.removeItem(at: directory)
    }

    // -----------------------------
    // Recording
    // -----------------------------
    private func run(device: AVCaptureDevice, countdown: Int, maxSeconds: TimeInterval, sampleRate: Double,
                     trimSilence: Bool) async {
        failure = nil
        guard await requestAccess() else { return }
        if let take { try? FileManager.default.removeItem(at: take.url) }
        take = nil
        self.maxSeconds = maxSeconds
        elapsed = 0
        level = 0

        // The session opens the microphone during the countdown, so that recording starts at once.
        let session = AVCaptureSession()
        let output = AVCaptureAudioFileOutput()
        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else { throw RecorderError.couldNotStart }
            session.addInput(input)
            output.audioSettings = Self.settings(sampleRate: sampleRate)
            output.maxRecordedDuration = CMTime(seconds: maxSeconds, preferredTimescale: 600)
            guard session.canAddOutput(output) else { throw RecorderError.couldNotStart }
            session.addOutput(output)
        } catch {
            failure = error.localizedDescription
            return
        }
        state = .countdown(max(countdown, 1))
        await Self.setRunning(session, true)

        for remaining in stride(from: countdown, through: 1, by: -1) {
            state = .countdown(remaining)
            try? await Task.sleep(for: .seconds(1))
            if Task.isCancelled { break }
        }
        if Task.isCancelled || !session.isRunning {
            if !Task.isCancelled { failure = RecorderError.couldNotStart.localizedDescription }
            await Self.setRunning(session, false)
            state = .idle
            return
        }

        let raw = directory.appending(path: "take-\(UUID().uuidString)-raw.wav")
        let finished = RecordingFinished()
        output.startRecording(to: raw, outputFileType: .wav, recordingDelegate: finished)
        self.output = output
        state = .recording
        while !finished.isDone, !Task.isCancelled {
            if let channel = output.connection(with: .audio)?.audioChannels.first {
                level = Self.normalized(decibels: Double(channel.averagePowerLevel))
            }
            let seconds = output.recordedDuration.seconds
            elapsed = seconds.isFinite ? seconds : 0
            try? await Task.sleep(for: .milliseconds(50))
        }
        if !finished.isDone { output.stopRecording() }
        let error = await finished.wait()
        self.output = nil
        await Self.setRunning(session, false)
        level = 0
        state = .idle
        if Task.isCancelled {
            try? FileManager.default.removeItem(at: raw)
            return
        }
        // Reaching the time limit ends the recording with an error that marks it as complete.
        if let error, (error as NSError).userInfo[AVErrorRecordingSuccessfullyFinishedKey] as? Bool != true {
            failure = error.localizedDescription
            try? FileManager.default.removeItem(at: raw)
            return
        }
        finish(raw: raw, trimSilence: trimSilence)
    }

    /// Start or stop the session off the main thread (`startRunning` blocks until the device is open).
    private static func setRunning(_ session: AVCaptureSession, _ running: Bool) async {
        let box = UncheckedSendable(value: session)
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                if running { box.value.startRunning() } else { box.value.stopRunning() }
                continuation.resume()
            }
        }
    }

    private func requestAccess() async -> Bool {
        access = MicrophoneAccess.current
        switch access {
        case .granted:
            return true
        case .undetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .audio)
            access = granted ? .granted : .denied
            return granted
        case .denied, .needsAppBundle:
            return false
        }
    }

    private func finish(raw: URL, trimSilence: Bool) {
        let final = directory.appending(path: raw.lastPathComponent.replacingOccurrences(of: "-raw", with: ""))
        do {
            let analysis = try SilenceTrimmer.process(input: raw, output: final, trim: trimSilence)
            try? FileManager.default.removeItem(at: raw)
            take = Take(url: final, duration: analysis.duration, peakDecibels: analysis.peakDecibels)
        } catch {
            try? FileManager.default.removeItem(at: final)
            failure = error.localizedDescription
            try? FileManager.default.removeItem(at: raw)
        }
    }

    private static func settings(sampleRate: Double) -> [String: Any] {
        [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
    }

    /// dBFS of the meter to 0...1 (-55 dB and below is 0).
    private static func normalized(decibels: Double) -> Double {
        min(1, max(0, (decibels + 55) / 50))
    }

    /// Remove recording directories of earlier runs whose process is gone (own user, not symlinks).
    private static func removeStaleDirectories() {
        let fileManager = FileManager.default
        let temporary = fileManager.temporaryDirectory
        let names = (try? fileManager.contentsOfDirectory(atPath: temporary.path)) ?? []
        for name in names where name.hasPrefix(directoryPrefix) {
            guard let pidText = name.dropFirst(directoryPrefix.count).split(separator: "-").first,
                  let pid = Int32(pidText), pid != getpid() else { continue }
            if kill(pid, 0) == 0 || errno == EPERM { continue }  // still running
            let path = temporary.appending(path: name).path
            guard let attributes = try? fileManager.attributesOfItem(atPath: path),  // does not follow symlinks
                  attributes[.type] as? FileAttributeType == .typeDirectory,
                  (attributes[.ownerAccountID] as? NSNumber)?.uint32Value == getuid() else { continue }
            try? fileManager.removeItem(atPath: path)
        }
    }
}

enum RecorderError: LocalizedError {
    case couldNotStart

    var errorDescription: String? { "The microphone could not start recording." }
}

/// Waits for AVCaptureFileOutput to finish writing; its delegate is called on another queue.
private final class RecordingFinished: NSObject, AVCaptureFileOutputRecordingDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    private var error: Error?
    private var continuation: CheckedContinuation<Error?, Never>?

    var isDone: Bool { lock.withLock { done } }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL,
                    from connections: [AVCaptureConnection], error: Error?) {
        let continuation: CheckedContinuation<Error?, Never>? = lock.withLock {
            done = true
            self.error = error
            defer { self.continuation = nil }
            return self.continuation
        }
        continuation?.resume(returning: error)
    }

    func wait() async -> Error? {
        await withCheckedContinuation { (continuation: CheckedContinuation<Error?, Never>) in
            let result: (Bool, Error?) = lock.withLock {
                if !done { self.continuation = continuation }
                return (done, error)
            }
            if result.0 { continuation.resume(returning: result.1) }
        }
    }
}

/// A value handed to another queue that the receiver uses exclusively.
private struct UncheckedSendable<Value>: @unchecked Sendable {
    let value: Value
}

/// Cuts leading and trailing silence of a recording, keeping a margin around the speech, and
/// measures it. Energy-based: frames well above the noise floor and near the peak are speech.
enum SilenceTrimmer {
    struct Analysis {
        var duration: TimeInterval
        var peakDecibels: Double
    }

    static let frameSeconds = 0.02
    static let marginSeconds = 0.25

    /// Write `input` to `output`, trimmed when `trim` and speech is found. Both are WAV files.
    static func process(input: URL, output: URL, trim: Bool) throws -> Analysis {
        let file = try AVAudioFile(forReading: input)
        let format = file.processingFormat  // float32, non-interleaved
        let frameCount = AVAudioFrameCount(file.length)
        guard frameCount > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw RecorderError.couldNotStart
        }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else { throw RecorderError.couldNotStart }
        let samples = UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength))
        let rate = format.sampleRate

        // Energy of each 20 ms frame, in dBFS
        let window = max(1, Int(rate * frameSeconds))
        let energies: [Double] = stride(from: 0, to: samples.count, by: window).map { start in
            let frame = samples[start..<min(samples.count, start + window)]
            let power = frame.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(frame.count)
            return 10 * log10(max(power, 1e-12))
        }
        let peak = energies.max() ?? -120

        var range = 0..<samples.count
        if trim, let speech = speechFrames(energies: energies, peak: peak) {
            let margin = Int(rate * marginSeconds)
            let start = max(0, speech.lowerBound * window - margin)
            let end = min(samples.count, speech.upperBound * window + margin)
            if end - start >= Int(rate * 0.5) { range = start..<end }  // keep everything if too little is left
        }

        let out = try AVAudioFile(forWriting: output, settings: file.fileFormat.settings,
                                  commonFormat: .pcmFormatFloat32, interleaved: false)
        let count = AVAudioFrameCount(range.count)
        guard let slice = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count),
              let target = slice.floatChannelData?[0] else { throw RecorderError.couldNotStart }
        target.update(from: channel + range.lowerBound, count: range.count)
        slice.frameLength = count
        try out.write(from: slice)
        return Analysis(duration: Double(range.count) / rate, peakDecibels: peak)
    }

    /// Frames from the first to the last one that counts as speech: within 35 dB of the peak
    /// and 8 dB above the noise floor (the 10th percentile of frame energies).
    private static func speechFrames(energies: [Double], peak: Double) -> Range<Int>? {
        guard energies.count > 2 else { return nil }
        let floor = energies.sorted()[energies.count / 10]
        let threshold = min(peak - 6, max(peak - 35, floor + 8))
        guard let first = energies.firstIndex(where: { $0 > threshold }),
              let last = energies.lastIndex(where: { $0 > threshold }) else { return nil }
        return first..<(last + 1)
    }
}
