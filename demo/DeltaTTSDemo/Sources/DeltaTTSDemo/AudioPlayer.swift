import AVFoundation
import Observation

/// Plays one sound (the output or a prompt preview) and reports whether it is playing.
@MainActor
@Observable
final class AudioPlayer: NSObject, AVAudioPlayerDelegate {
    private(set) var isPlaying = false
    private(set) var duration: TimeInterval = 0
    @ObservationIgnored private var player: AVAudioPlayer?

    /// Playback position; read it from a `TimelineView` while playing.
    var currentTime: TimeInterval { player?.currentTime ?? 0 }

    func load(data: Data) throws {
        stop()
        let player = try AVAudioPlayer(data: data)
        attach(player)
    }

    func load(url: URL) throws {
        stop()
        let player = try AVAudioPlayer(contentsOf: url)
        attach(player)
    }

    func unload() {
        stop()
        player = nil
        duration = 0
    }

    func play() {
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

    private func attach(_ player: AVAudioPlayer) {
        player.delegate = self
        player.prepareToPlay()
        self.player = player
        duration = player.duration
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.isPlaying = false }
    }
}
