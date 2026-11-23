import AVFoundation
import Foundation
import Observation

/// Which prompt the user picked: a preset of the config or a file of their own.
enum PromptSelection: Hashable {
    case none
    case preset(UUID)
    case file
}

/// Duration and sample rate of a prompt file, as AVFoundation reads them.
struct PromptInfo: Equatable {
    var duration: TimeInterval
    var sampleRate: Double
}

/// State of the server as the health polling sees it.
enum ServerState: Equatable {
    case unknown
    case unreachable(String)
    case loading
    case ready
    case failed(String)
}

/// The last synthesized output.
struct SynthesisOutput {
    var wav: Data
    var response: SynthesisResponse
    var text: String
    var date = Date()
}

struct AlertItem: Identifiable {
    let id = UUID()
    var title: String
    var message: String
}

/// App state shared by the main window and the Settings scene.
@MainActor
@Observable
final class AppModel {
    /// The persisted config (saved to Application Support shortly after each change).
    var config: DemoConfig {
        didSet { if config != oldValue { scheduleSave() } }
    }

    // Inputs
    private(set) var selection: PromptSelection = .none
    private(set) var customPromptURL: URL?
    var transcript = ""
    var text = "" {
        didSet { if !text.isEmpty { attemptedWithoutInput = false } }
    }

    // Server
    private(set) var health: Health?
    private(set) var healthError: String?
    let serverProcess = ServerProcess()

    // Synthesis
    private(set) var isSynthesizing = false
    private(set) var output: SynthesisOutput?
    private(set) var serverNotices: [Notice] = []
    private var dismissedNotices: Set<String> = []
    private var attemptedWithoutInput = false
    var alert: AlertItem?
    let player = AudioPlayer()
    let promptPlayer = AudioPlayer()

    /// The voice-cloning notice was acknowledged with "Don't show again".
    var consentAcknowledged: Bool {
        didSet { UserDefaults.standard.set(consentAcknowledged, forKey: Self.consentKey) }
    }

    nonisolated static let consentKey = "voiceCloningNoticeAcknowledged"
    nonisolated static let maxUploadBytes = 18 * 1024 * 1024  // MAX_UPLOAD_BYTES of demo/server.py

    @ObservationIgnored private var saveTask: Task<Void, Never>?
    @ObservationIgnored private var promptInfoCache: [String: Result<PromptInfo, PromptError>] = [:]

    init() {
        var config = DemoConfig()
        var loadError: String?
        do {
            if let user = try ConfigStore.loadUser() {
                config = user
            } else if let repo = ConfigStore.builtFromRepository() {
                config = try ConfigStore.loadRepoDefaults(repoPath: repo)
            }
        } catch {
            loadError = "\(ConfigStore.userConfigURL.path): \(error.localizedDescription) The defaults are used instead."
            if let repo = ConfigStore.builtFromRepository(), let defaults = try? ConfigStore.loadRepoDefaults(repoPath: repo) {
                config = defaults
            }
        }
        if config.app.repoPath.isEmpty, let repo = ConfigStore.builtFromRepository() {
            config.app.repoPath = repo
        }
        self.config = config
        consentAcknowledged = UserDefaults.standard.bool(forKey: Self.consentKey)
        if let preset = config.presets.first {
            selection = .preset(preset.id)
            transcript = preset.transcript
        }
        if let loadError {
            alert = AlertItem(title: "Cannot read your settings", message: loadError)
        }
    }

    // -----------------------------
    // Prompt
    // -----------------------------
    var selectedPreset: DemoConfig.Preset? {
        guard case .preset(let id) = selection else { return nil }
        return config.presets.first { $0.id == id }
    }

    var promptURL: URL? {
        switch selection {
        case .none: nil
        case .preset: selectedPreset.map { config.resolve($0.wav) }
        case .file: customPromptURL
        }
    }

    var promptName: String {
        switch selection {
        case .none: "No prompt"
        case .preset: selectedPreset?.name ?? "No prompt"
        case .file: customPromptURL?.lastPathComponent ?? "No prompt"
        }
    }

    func select(_ selection: PromptSelection) {
        guard selection != self.selection else { return }
        promptPlayer.unload()
        self.selection = selection
        switch selection {
        case .preset: transcript = selectedPreset?.transcript ?? ""
        case .file, .none: transcript = ""
        }
    }

    func useCustomPrompt(_ url: URL) {
        promptPlayer.unload()
        customPromptURL = url
        selection = .file
        transcript = ""
    }

    /// Keep the selection valid after the presets change (edits in Settings, reset).
    func presetsChanged() {
        if case .preset = selection, selectedPreset == nil {
            select(config.presets.first.map { .preset($0.id) } ?? .none)
        }
    }

    /// Duration and sample rate of the current prompt, or why it cannot be read.
    var promptInfo: Result<PromptInfo, PromptError>? {
        guard let url = promptURL else { return nil }
        return info(for: url)
    }

    private func info(for url: URL) -> Result<PromptInfo, PromptError> {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        guard let attributes else { return .failure(.notFound(url.path)) }
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let key = "\(url.path)|\(modified)"
        if let cached = promptInfoCache[key] { return cached }
        let result: Result<PromptInfo, PromptError>
        do {
            let file = try AVAudioFile(forReading: url)
            let rate = file.processingFormat.sampleRate
            result = .success(PromptInfo(duration: Double(file.length) / rate, sampleRate: file.fileFormat.sampleRate))
        } catch {
            result = .failure(.unreadable(error.localizedDescription))
        }
        promptInfoCache[key] = result
        return result
    }

    func togglePromptPreview() {
        if promptPlayer.isPlaying {
            promptPlayer.stop()
            return
        }
        guard let url = promptURL else { return }
        do {
            try promptPlayer.load(url: url)
            player.stop()
            promptPlayer.play()
        } catch {
            alert = AlertItem(title: "Cannot play the prompt", message: error.localizedDescription)
        }
    }

    // -----------------------------
    // Server
    // -----------------------------
    var serverState: ServerState {
        if let healthError { return .unreachable(healthError) }
        guard let health else { return .unknown }
        switch health.status {
        case "ready": return .ready
        case "loading": return .loading
        default: return .failed(health.error ?? "unknown error")
        }
    }

    private var client: ServerClient? {
        config.serverURL.map { ServerClient(baseURL: $0) }
    }

    /// Poll /health until the task is cancelled: often while the server is down or loading.
    func pollHealth() async {
        while !Task.isCancelled {
            await refreshHealth()
            try? await Task.sleep(for: .seconds(serverState == .ready ? 5 : 1))
        }
    }

    func refreshHealth() async {
        guard let client else {
            health = nil
            healthError = "invalid server URL"
            return
        }
        do {
            health = try await client.health()
            healthError = nil
        } catch let error as ServerError {
            health = nil
            if case .unreachable = error { healthError = "not reachable" } else { healthError = error.localizedDescription }
        } catch {
            health = nil
            healthError = error.localizedDescription
        }
    }

    func startServer() {
        do {
            saveTask?.cancel()
            try ConfigStore.save(config)  // the server reads this file
            try serverProcess.start(repoPath: config.app.repoPath, uvPath: config.app.uvPath,
                                    configURL: ConfigStore.userConfigURL, dryRun: config.app.launchDryRun)
        } catch {
            alert = AlertItem(title: "Cannot start the server", message: error.localizedDescription)
        }
    }

    // -----------------------------
    // Synthesis
    // -----------------------------
    /// Why synthesis cannot start now, if it cannot.
    var blockingReason: String? {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter the text to synthesize." }
        guard promptURL != nil else { return "Choose a voice prompt." }
        if case .failure = promptInfo { return "The prompt audio cannot be read." }
        switch serverState {
        case .unknown: return "Checking the server..."
        case .unreachable: return "The server is not reachable."
        case .loading: return "The server is still loading the model."
        case .failed: return "The server failed to load the model."
        case .ready: return nil
        }
    }

    func synthesize() async {
        guard !isSynthesizing else { return }
        guard blockingReason == nil, let promptURL, let client else {
            attemptedWithoutInput = true
            return
        }
        isSynthesizing = true
        defer { isSynthesizing = false }
        player.stop()
        promptPlayer.stop()
        do {
            let audio = try Data(contentsOf: promptURL)
            guard audio.count <= Self.maxUploadBytes else {
                throw PromptError.tooLarge(audio.count)
            }
            let s = config.synthesis, p = config.processing
            let request = SynthesisRequest(
                text: text, promptText: transcript,
                promptWavBase64: audio.base64EncodedString(), promptWavName: promptURL.lastPathComponent,
                lengthScale: s.lengthScale, numSteps: s.numSteps, temperature: s.temperature, topP: s.topP,
                tShift: s.tShift, compute: s.compute,
                enhancePrompt: p.enhancePrompt, trimPrompt: p.trimPrompt, trimOutput: p.trimOutput,
                warnings: config.warnings)
            let response = try await client.synthesize(request)
            guard let wav = Data(base64Encoded: response.audioWavBase64) else {
                throw ServerError.invalidResponse("the audio is not valid base64")
            }
            try player.load(data: wav)
            output = SynthesisOutput(wav: wav, response: response, text: text)
            serverNotices = response.warnings
            attemptedWithoutInput = false
            player.play()
        } catch let error as ServerError {
            serverNotices = error.notices
            alert = AlertItem(title: "Synthesis failed", message: error.localizedDescription)
        } catch {
            alert = AlertItem(title: "Synthesis failed", message: error.localizedDescription)
        }
    }

    // -----------------------------
    // Notices
    // -----------------------------
    /// The app's own checks, recomputed from the current state.
    var clientNotices: [Notice] {
        var notices: [Notice] = []
        if !consentAcknowledged {
            notices.append(Notice(level: .info, code: "consent", message:
                "Voice cloning: use only prompts from speakers who agreed to it, and do not present "
                + "synthesized speech as real recordings of them."))
        }
        switch serverState {
        case .unknown, .ready: break
        case .unreachable(let reason):
            notices.append(Notice(level: .warning, code: "server_unreachable", message:
                "The server at \(config.serverURL?.absoluteString ?? "(invalid URL)") is \(reason). Start it with "
                + "\"Start Server\" or `uv run python demo/server.py` in the repository."))
        case .loading:
            notices.append(Notice(level: .info, code: "server_loading", message: "The server is loading the model..."))
        case .failed(let error):
            notices.append(Notice(level: .error, code: "server_failed", message: "The server failed to load the model: \(error)"))
        }
        if health?.dryRun == true {
            notices.append(Notice(level: .info, code: "dry_run", message:
                "The server runs in dry-run mode: outputs are tones as long as the estimated speech, not speech."))
        }

        let limits = config.warnings
        if attemptedWithoutInput && text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            notices.append(Notice(level: .warning, code: "text_empty", message: "Enter the text to synthesize."))
        }
        if Double(text.count) > limits.maxTextChars {
            notices.append(Notice(level: .warning, code: "text_long", message:
                "The text has \(text.count) characters (more than \(Int(limits.maxTextChars))). DELTA-TTS generates "
                + "the whole utterance at once; consider splitting long text into sentences."))
        }
        if promptURL == nil {
            if attemptedWithoutInput {
                notices.append(Notice(level: .warning, code: "prompt_missing", message: "Choose a voice prompt."))
            }
        } else {
            if transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                notices.append(Notice(level: .warning, code: "transcript_missing", message:
                    "No prompt transcript. DELTA-TTS expects the words spoken in the prompt, and the output length "
                    + "falls back to a default speaking rate; expect lower quality."))
            }
            switch promptInfo {
            case .success(let info): notices += promptChecks(info)
            case .failure(let error):
                notices.append(Notice(level: .error, code: "prompt_unreadable", message: error.localizedDescription))
            case nil: break
            }
        }
        if config.synthesis.compute == .fast {
            notices.append(Notice(level: .info, code: "fast_mode", message:
                "Fast mode runs DELTA-TTS in float16: outputs differ numerically from exact mode, and an utterance "
                + "that overflows is finished in float32."))
        }
        return notices
    }

    private func promptChecks(_ info: PromptInfo) -> [Notice] {
        let limits = config.warnings
        var notices: [Notice] = []
        let seconds = String(format: "%.1f", info.duration)
        if info.sampleRate < 16000 {
            notices.append(Notice(level: .error, code: "prompt_low_sample_rate", message:
                "The prompt is sampled at \(Int(info.sampleRate)) Hz; CosyVoice 3 needs at least 16000 Hz."))
        } else if info.sampleRate < limits.minPromptSampleRate {
            notices.append(Notice(level: .warning, code: "prompt_low_sample_rate", message:
                "The prompt is sampled at \(Int(info.sampleRate)) Hz (below \(Int(limits.minPromptSampleRate)) Hz); "
                + "the cloned voice may sound muffled."))
        }
        if info.duration < limits.minPromptSeconds {
            notices.append(Notice(level: .warning, code: "prompt_short", message:
                "The prompt is \(seconds) s long (under \(limits.minPromptSeconds.formatted()) s); a short prompt gives "
                + "a less accurate voice and speaking rate."))
        } else if info.duration > limits.maxPromptSeconds {
            notices.append(Notice(level: info.duration > 30 && !config.processing.trimPrompt ? .error : .warning,
                                  code: "prompt_long", message:
                "The prompt is \(seconds) s long (over \(limits.maxPromptSeconds.formatted()) s); CosyVoice 3 accepts "
                + "at most 30 s, and a long prompt slows synthesis."))
        }
        return notices
    }

    /// Client checks and the server's notices of the last request, minus dismissed ones. A
    /// server notice is hidden when a client check reports the same thing (same code).
    var visibleNotices: [Notice] {
        let client = clientNotices
        let codes = Set(client.map(\.code))
        let server = serverNotices.filter { !codes.contains($0.code) }
        var seen = Set<String>()
        return (client + server)
            .filter { !dismissedNotices.contains($0.id) && seen.insert($0.id).inserted }
            .enumerated()
            .sorted { ($0.element.level, $0.offset) < ($1.element.level, $1.offset) }
            .map(\.element)
    }

    func dismiss(_ notice: Notice) {
        dismissedNotices.insert(notice.id)
    }

    // -----------------------------
    // Config
    // -----------------------------
    /// Replace the config with demo/config.json of the repository, keeping the app's paths.
    func resetToDefaults() {
        do {
            var fresh = try ConfigStore.loadRepoDefaults(repoPath: config.app.repoPath)
            fresh.app.uvPath = config.app.uvPath
            fresh.app.launchDryRun = config.app.launchDryRun
            config = fresh
            presetsChanged()
        } catch {
            alert = AlertItem(title: "Cannot reset to defaults", message: error.localizedDescription)
        }
    }

    private func scheduleSave() {
        saveTask?.cancel()
        saveTask = Task {
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled else { return }
            do {
                try ConfigStore.save(config)
            } catch {
                alert = AlertItem(title: "Cannot save the settings", message: error.localizedDescription)
            }
        }
    }
}

enum PromptError: LocalizedError, Equatable {
    case notFound(String)
    case unreadable(String)
    case tooLarge(Int)

    var errorDescription: String? {
        switch self {
        case .notFound(let path): "Prompt audio not found: \(path)"
        case .unreadable(let reason): "Cannot read the prompt audio: \(reason)"
        case .tooLarge(let bytes):
            "The prompt file is \(bytes / 1_048_576) MB; the server accepts at most \(AppModel.maxUploadBytes / 1_048_576) MB."
        }
    }
}
