import AVFoundation
import Observation

/// Plays one sound (the output or the recording) on a chosen output device or the system default,
/// and reports whether it is playing.
///
/// A chosen device is set as the player's `currentDevice`. An AVAudioPlayer keeps the device it
/// started on, so with the system default the player is re-created when the default has changed
/// (headphones plugged in, Bluetooth connected): before the next playback, and during playback
/// through `outputDeviceChanged`, resuming where it was.
@MainActor
@Observable
final class AudioPlayer: NSObject, AVAudioPlayerDelegate {
    private enum Source {
        case data(Data)
        case url(URL)
    }

    private(set) var isPlaying = false
    private(set) var duration: TimeInterval = 0
    private(set) var timesFinished = 0  // playbacks of the loaded sound that reached the end
    private(set) var atEnd = false  // the last playback reached the end (AVAudioPlayer then reports 0)
    @ObservationIgnored private var player: AVAudioPlayer?
    @ObservationIgnored private var source: Source?
    @ObservationIgnored private var device = AudioDeviceID(0)  // the default output when `player` was made
    @ObservationIgnored private var routedDevice: String?  // UID of the chosen output, nil for the system default

    /// Playback position; read it from a `TimelineView` while playing. At the end of a playback it
    /// stays at `duration` until the next one starts.
    var currentTime: TimeInterval {
        if atEnd { return duration }
        return isPlaying ? min(player?.currentTime ?? 0, duration) : 0
    }

    /// `currentTime` as a fraction of `duration`, 0...1.
    var progress: Double {
        duration > 0 ? min(1, max(0, currentTime / duration)) : 0
    }

    func load(data: Data) throws {
        stop()
        timesFinished = 0
        atEnd = false
        source = .data(data)
        try makePlayer()
    }

    func load(url: URL) throws {
        stop()
        timesFinished = 0
        atEnd = false
        source = .url(url)
        try makePlayer()
    }

    func unload() {
        stop()
        player = nil
        source = nil
        duration = 0
        timesFinished = 0
        atEnd = false
    }

    /// Play from the start, or from `time`.
    func play(from time: TimeInterval = 0) {
        if routedDevice == nil, device != AudioDevices.defaultDevice(output: true) {
            try? makePlayer()
        }
        guard let player else { return }
        player.currentTime = min(max(0, time), max(0, duration - 0.01))
        atEnd = false
        isPlaying = player.play()
    }

    func stop() {
        player?.stop()
        player?.currentTime = 0
        isPlaying = false
        atEnd = false
    }

    func toggle() {
        isPlaying ? stop() : play()
    }

    /// Play on the output device with this UID, or on the system default when nil.
    func route(to deviceUID: String?) {
        guard deviceUID != routedDevice else { return }
        routedDevice = deviceUID
        outputDeviceChanged()
    }

    /// Move playback to the current output device, continuing where it was.
    func outputDeviceChanged() {
        guard player != nil else { return }
        let wasPlaying = isPlaying
        let position = player?.currentTime ?? 0
        guard (try? makePlayer()) != nil, let player else { return }
        if wasPlaying {
            player.currentTime = position
            isPlaying = player.play()
        }
    }

    private func makePlayer() throws {
        player?.delegate = nil
        player?.stop()
        let player: AVAudioPlayer
        switch source {
        case .data(let data): player = try AVAudioPlayer(data: data)
        case .url(let url): player = try AVAudioPlayer(contentsOf: url)
        case nil: return
        }
        player.delegate = self
        if let routedDevice { player.currentDevice = routedDevice }
        player.prepareToPlay()
        self.player = player
        device = AudioDevices.defaultDevice(output: true)
        duration = player.duration
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.isPlaying = false
            self.atEnd = true
            self.timesFinished += 1
        }
    }
}
