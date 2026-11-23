import AVFoundation
import Observation

/// Plays one sound (the output or the recording) on the system default output device, and
/// reports whether it is playing.
///
/// An AVAudioPlayer keeps the device it started on, so the player is re-created when the default
/// output has changed (headphones plugged in, Bluetooth connected): before the next playback,
/// and during playback through `outputDeviceChanged`.
@MainActor
@Observable
final class AudioPlayer: NSObject, AVAudioPlayerDelegate {
    private enum Source {
        case data(Data)
        case url(URL)
    }

    private(set) var isPlaying = false
    private(set) var duration: TimeInterval = 0
    @ObservationIgnored private var player: AVAudioPlayer?
    @ObservationIgnored private var source: Source?
    @ObservationIgnored private var device = AudioDeviceID(0)  // the default output when `player` was made

    /// Playback position; read it from a `TimelineView` while playing.
    var currentTime: TimeInterval { player?.currentTime ?? 0 }

    func load(data: Data) throws {
        stop()
        source = .data(data)
        try makePlayer()
    }

    func load(url: URL) throws {
        stop()
        source = .url(url)
        try makePlayer()
    }

    func unload() {
        stop()
        player = nil
        source = nil
        duration = 0
    }

    func play() {
        if device != AudioDevices.defaultDevice(output: true) {
            try? makePlayer()
        }
        guard let player else { return }
        isPlaying = player.play()
    }

    func stop() {
        player?.stop()
        player?.currentTime = 0
        isPlaying = false
    }

    func toggle() {
        isPlaying ? stop() : play()
    }

    /// Move playback to the new default output, continuing where it was.
    func outputDeviceChanged() {
        guard player != nil, isPlaying else { return }
        let position = player?.currentTime ?? 0
        guard (try? makePlayer()) != nil, let player else { return }
        player.currentTime = position
        isPlaying = player.play()
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
        player.prepareToPlay()
        self.player = player
        device = AudioDevices.defaultDevice(output: true)
        duration = player.duration
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.isPlaying = false }
    }
}
