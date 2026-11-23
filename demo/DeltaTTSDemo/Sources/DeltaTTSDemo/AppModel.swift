import AppKit
import AVFoundation
import Foundation
import Observation

/// Whose voice is the prompt: the visitor's recording or a sample voice (a preset).
enum PromptSelection: Hashable {
    case recording
    case preset(UUID)
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

/// The last generated output, and the inputs it was made from.
struct SynthesisOutput {
    var wav: Data
    var response: SynthesisResponse
    var inputKey: String
}

/// The line under the header that says what to do now.
struct Instruction: Equatable {
    var text: String
    var level: Notice.Level
    var systemImage: String
}

struct AlertItem: Identifiable {
    let id = UUID()
    var title: String
    var message: String
}

/// An extra button on a notice.
struct NoticeAction {
    var title: String
    var perform: () -> Void
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
    private(set) var selection = PromptSelection.recording
    var text = "" {
        didSet {
            if !text.isEmpty { attemptedWithoutInput = false }
            if text != oldValue { generationError = nil }
            if text != oldValue, text != config.demo.defaultText { readyForNextVisitor = false }
        }
    }

    // Server
    private(set) var health: Health?
    private(set) var healthError: String?
    let serverProcess = ServerProcess()

    // Recording and playback
    let recorder = VoiceRecorder()
    let player = AudioPlayer()  // the generated voice
    let promptPlayer = AudioPlayer()  // the recording or a sample voice
    let audioDevices = AudioDevices()
    private(set) var missingMicrophone: String?  // the chosen microphone was not connected at the last recording
    private(set) var noMicrophone = false
    private(set) var missingOutput: String?  // the chosen output device is not connected
    private(set) var outputNotice: Notice?  // shown for a few seconds after the output changed by itself

    // Generation
    private(set) var isSynthesizing = false
    private(set) var output: SynthesisOutput?
    private(set) var generationError: String?
    private(set) var lastTiming: String?  // for staff, in Settings; never on the main screen

    /// Processing of the prompt on the server (/prepare), started as soon as a recording stops or a
    /// sample voice is chosen, so that Generate only has to generate.
    enum PrepareState: Equatable {
        case idle
        case preparing
        case ready(id: String, seconds: Double)
        case failed(String)
    }

    private(set) var prepareState = PrepareState.idle
    /// The server's progress of the generation in flight, polled while it runs.
    private(set) var generationProgress: ProgressSnapshot?
    /// Its fraction, never going backwards within one generation; nil while indeterminate.
    private(set) var generationFraction: Double?
    @ObservationIgnored private var prepareTask: Task<Void, Never>?
    @ObservationIgnored private var synthesisTask: Task<Void, Never>?
    @ObservationIgnored private var progressTask: Task<Void, Never>?
    private(set) var prepareNotices: [Notice] = []
    @ObservationIgnored private var prepareToken = UUID()  // tags a prepare; a later one makes it stale
    @ObservationIgnored private var preparedKey = ""  // the prompt, transcript and processing it was for
    private(set) var serverNotices: [Notice] = []
    private(set) var isResetting = false
    private(set) var readyForNextVisitor = false  // shown briefly after Next visitor
    private var resetNotice: Notice?
    private var dismissedNotices: Set<String> = []
    private var attemptedWithoutInput = false
    var alert: AlertItem?

    /// The voice-cloning notice was acknowledged with "Don't show again".
    var consentAcknowledged: Bool {
        didSet { UserDefaults.standard.set(consentAcknowledged, forKey: Self.consentKey) }
    }

    nonisolated static let consentKey = "voiceCloningNoticeAcknowledged"
    nonisolated static let maxUploadBytes = 18 * 1024 * 1024  // MAX_UPLOAD_BYTES of demo/server.py

    @ObservationIgnored private var visitor = UUID()  // replaced by Next visitor; late results of the previous visitor are dropped
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
        text = config.demo.defaultText
        recorder.onTake = { [weak self] _ in self?.preparePrompt() }
        audioDevices.onDefaultOutputChange = { [weak self] in self?.defaultOutputChanged() }
        audioDevices.onChange = { [weak self] in self?.applyOutputRoute() }
        applyOutputRoute()
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

    /// The prompt audio: the recording, or the sample voice's file.
    var promptURL: URL? {
        switch selection {
        case .recording: recorder.take?.url
        case .preset: selectedPreset.map { config.resolve($0.wav) }
        }
    }

    /// The words spoken in the prompt: the sentence read aloud, or the sample voice's transcript.
    var transcript: String {
        switch selection {
        case .recording: config.demo.defaultTranscript
        case .preset: selectedPreset?.transcript ?? ""
        }
    }

    func useSampleVoice(_ id: UUID) {
        promptPlayer.unload()
        selection = .preset(id)
        preparePrompt()
    }

    func useOwnVoice() {
        promptPlayer.unload()
        selection = .recording
        preparePrompt()
    }

    /// The prompt, transcript and processing a prepared prompt_id stands for.
    private var promptKey: String {
        let p = config.processing
        return "\(promptURL?.path ?? "")|\(transcript)|\(p.enhancePrompt)\(p.trimPrompt)\(p.trimOutput)"
    }

    /// Start /prepare for the current prompt, in the background. A newer call, a new recording or
    /// Next visitor makes it stale: its response is then dropped.
    func preparePrompt() {
        let token = UUID()
        prepareToken = token
        prepareNotices = []
        guard let promptURL, let client else {
            prepareState = .idle
            return
        }
        preparedKey = promptKey
        prepareState = .preparing
        let p = config.processing
        let promptText = transcript, warnings = config.warnings
        let name = selection == .recording ? "recording.wav" : promptURL.lastPathComponent
        prepareTask?.cancel()
        prepareTask = Task {
            do {
                let audio = try Data(contentsOf: promptURL)
                guard audio.count <= Self.maxUploadBytes else { throw PromptError.tooLarge(audio.count) }
                let request = PrepareRequest(promptText: promptText, promptWavBase64: audio.base64EncodedString(),
                                             promptWavName: name, enhancePrompt: p.enhancePrompt,
                                             trimPrompt: p.trimPrompt, trimOutput: p.trimOutput, warnings: warnings)
                let response = try await client.prepare(request)
                guard token == prepareToken else { return }
                prepareState = .ready(id: response.promptId, seconds: response.durationSeconds)
                prepareNotices = response.warnings.filter { $0.level != .info }
            } catch {
                guard token == prepareToken else { return }
                let notices = (error as? ServerError)?.notices.filter { $0.level != .info } ?? []
                prepareNotices = notices.filter { $0.level != .error }
                let reason = notices.first { $0.level == .error }?.message ?? error.localizedDescription
                prepareState = .failed(reason)
            }
        }
    }

    /// Forget any prepare in progress or done (a new recording, Next visitor).
    private func dropPrepare() {
        prepareTask?.cancel()
        prepareToken = UUID()
        prepareState = .idle
        prepareNotices = []
        preparedKey = ""
    }

    /// Why the last /prepare failed (for staff, in a tooltip).
    var prepareFailure: String? {
        if case .failed(let reason) = prepareState { return reason }
        return nil
    }

    /// The prompt_id to send instead of the audio, when the current prompt was prepared as it is now.
    private var preparedID: String? {
        guard case .ready(let id, _) = prepareState, preparedKey == promptKey else { return nil }
        return id
    }

    /// Keep the selection valid after the presets change (edits in Settings, reset).
    func presetsChanged() {
        if case .preset = selection, selectedPreset == nil { useOwnVoice() }
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

    /// Play or stop the recording (or the sample voice).
    func togglePromptPlayback() {
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
            alert = AlertItem(title: "Cannot play the recording", message: error.localizedDescription)
        }
    }

    // -----------------------------
    // Recording
    // -----------------------------
    var canRecord: Bool {
        recorder.access != .denied && recorder.access != .needsAppBundle && !isSynthesizing && !isResetting
    }

    func startRecording() {
        readyForNextVisitor = false
        dropPrepare()
        promptPlayer.unload()
        player.stop()
        selection = .recording
        generationError = nil
        guard let device = inputDeviceForRecording() else {
            noMicrophone = true
            return
        }
        noMicrophone = false
        let settings = config.recording
        recorder.start(device: device, countdown: settings.countdownSeconds, maxSeconds: settings.maxSeconds,
                       sampleRate: settings.sampleRate, trimSilence: settings.trimSilence)
    }

    /// The chosen microphone, or the system default input when none is chosen or the chosen one
    /// is not connected.
    private func inputDeviceForRecording() -> AVCaptureDevice? {
        missingMicrophone = nil
        let chosen = config.recording.inputDeviceUid
        if !chosen.isEmpty {
            if let device = audioDevices.captureDevice(uniqueID: chosen) { return device }
            missingMicrophone = config.recording.inputDeviceName.isEmpty ? "The chosen microphone" : config.recording.inputDeviceName
        }
        return AVCaptureDevice.default(for: .audio)
    }

    /// Record from `device`, or from the system default input when nil.
    func chooseMicrophone(_ device: InputDevice?) {
        config.recording.inputDeviceUid = device?.id ?? ""
        config.recording.inputDeviceName = device?.name ?? ""
        missingMicrophone = nil
    }

    /// "System default (MacBook Pro Microphone)", the chosen microphone, or "... (not connected)".
    var microphoneLabel: String {
        let chosen = config.recording.inputDeviceUid
        if chosen.isEmpty {
            return "System default" + (audioDevices.defaultInputName.map { " (\($0))" } ?? "")
        }
        if let device = audioDevices.inputs.first(where: { $0.id == chosen }) { return device.name }
        let name = config.recording.inputDeviceName.isEmpty ? "Chosen microphone" : config.recording.inputDeviceName
        return "\(name) (not connected)"
    }

    /// "Output: MacBook Pro Speakers (system default)", or the chosen device.
    var outputLabel: String {
        "Output: \(outputName)"
    }

    /// The output device in use: the chosen one, or the system default.
    var outputName: String {
        let chosen = config.app.outputDeviceUid
        if !chosen.isEmpty, let device = audioDevices.outputs.first(where: { $0.id == chosen }) { return device.name }
        return "\(audioDevices.defaultOutputName ?? "unknown device") (system default)"
    }

    /// Play on `device`, or on the system default output when nil.
    func chooseOutput(_ device: OutputDevice?) {
        config.app.outputDeviceUid = device?.id ?? ""
        config.app.outputDeviceName = device?.name ?? ""
        outputNotice = nil
        applyOutputRoute()
    }

    /// The system default output changed (e.g. headphones plugged in): a chosen device gives way
    /// to it, so that the sound follows what was just connected.
    private func defaultOutputChanged() {
        if !config.app.outputDeviceUid.isEmpty {
            config.app.outputDeviceUid = ""
            config.app.outputDeviceName = ""
            let notice = Notice(level: .info, code: "output_switched", message:
                "Output switched to \(audioDevices.defaultOutputName ?? "the new default") (system default).")
            outputNotice = notice
            Task {
                try? await Task.sleep(for: .seconds(6))
                if outputNotice == notice { outputNotice = nil }
            }
        }
        applyOutputRoute()
        player.outputDeviceChanged()
        promptPlayer.outputDeviceChanged()
    }

    /// Route both players to the chosen output when it is connected, else to the system default.
    private func applyOutputRoute() {
        let chosen = config.app.outputDeviceUid
        let connected = !chosen.isEmpty && audioDevices.outputs.contains { $0.id == chosen }
        missingOutput = !chosen.isEmpty && !connected
            ? (config.app.outputDeviceName.isEmpty ? "The chosen output" : config.app.outputDeviceName) : nil
        player.route(to: connected ? chosen : nil)
        promptPlayer.route(to: connected ? chosen : nil)
    }

    /// The compute mode of both the requests and the server's startup (one setting).
    var computeMode: ComputeMode {
        get { config.synthesis.compute }
        set {
            config.synthesis.compute = newValue
            config.server.compute = newValue
        }
    }

    func stopRecording() {
        recorder.stop()
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
    // Generation
    // -----------------------------
    /// Identifies the inputs of a generation, to tell whether the output is still current.
    private var inputKey: String {
        let prompt = switch selection {
        case .recording: recorder.take?.url.path ?? ""
        case .preset(let id): id.uuidString
        }
        return "\(prompt)|\(transcript)|\(text)"
    }

    var outputIsCurrent: Bool {
        output?.inputKey == inputKey
    }

    /// Why generation cannot start now, in plain words, if it cannot.
    var blockingReason: String? {
        if recorder.isCapturing { return "Finish the recording first." }
        guard promptURL != nil else { return "Record your voice first." }
        if case .failure = promptInfo { return "The recording cannot be read. Please record again." }
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Type what your voice should say." }
        switch serverState {
        case .unknown, .loading: return "The voice engine is starting. Please wait."
        case .unreachable, .failed: return "The voice engine is not running. Please ask a staff member."
        case .ready: return nil
        }
    }

    /// Generate in a task the model owns, so that Next visitor can cancel it.
    func generate() {
        guard !isSynthesizing else { return }
        synthesisTask = Task { await synthesize() }
    }

    /// "Preparing your voice...", "Shaping your voice...", ... for the generation in flight.
    var generationStageLine: String {
        guard let progress = generationProgress, progress.state != "idle" else { return "Generating..." }
        if progress.state == "preparing" || progress.stage == "prompt" { return "Preparing your voice..." }
        switch progress.stage {
        case "tokens": return "Shaping your voice..."
        case "acoustic": return "Rendering audio..."
        case "vocoder": return "Almost done..."
        default: return "Generating..."
        }
    }

    /// Poll GET /progress every 100 ms while a generation is in flight.
    private func pollProgress(client: ServerClient) async {
        while !Task.isCancelled, isSynthesizing {
            if let snapshot = try? await client.progress(), !Task.isCancelled, isSynthesizing {
                generationProgress = snapshot
                if snapshot.state == "generating", let fraction = snapshot.fraction {
                    generationFraction = max(generationFraction ?? 0, fraction)
                }
            }
            try? await Task.sleep(for: .milliseconds(100))
        }
    }

    func synthesize() async {
        guard !isSynthesizing else { return }
        guard blockingReason == nil, let promptURL, let client else {
            attemptedWithoutInput = true
            return
        }
        let visitor = self.visitor
        let key = inputKey
        isSynthesizing = true
        generationError = nil
        generationProgress = nil
        generationFraction = nil
        progressTask = Task { await pollProgress(client: client) }
        defer {
            isSynthesizing = false
            progressTask?.cancel()
            generationProgress = nil
            generationFraction = nil
        }
        player.stop()
        promptPlayer.stop()
        do {
            let s = config.synthesis, p = config.processing
            var request = SynthesisRequest(
                text: text, promptText: transcript,
                lengthScale: s.lengthScale, numSteps: s.numSteps, temperature: s.temperature, topP: s.topP,
                tShift: s.tShift, compute: s.compute,
                enhancePrompt: p.enhancePrompt, trimPrompt: p.trimPrompt, trimOutput: p.trimOutput,
                warnings: config.warnings)
            let name = selection == .recording ? "recording.wav" : promptURL.lastPathComponent
            func attachAudio() throws {
                let audio = try Data(contentsOf: promptURL)
                guard audio.count <= Self.maxUploadBytes else { throw PromptError.tooLarge(audio.count) }
                request.promptId = nil
                request.promptWavBase64 = audio.base64EncodedString()
                request.promptWavName = name
            }
            // Prepared: send the id. Still preparing or not prepared: send the audio; the server
            // serializes it after a running /prepare, whose cached features it then reuses.
            if let id = preparedID {
                request.promptId = id
            } else {
                try attachAudio()
            }
            let response: SynthesisResponse
            do {
                response = try await client.synthesize(request)
            } catch let error as ServerError where error.code == "prompt_unknown" {
                try attachAudio()  // the server no longer knows the id (restarted, reset): send the audio
                response = try await client.synthesize(request)
            }
            guard visitor == self.visitor else { return }  // Next visitor was pressed meanwhile
            guard let wav = Data(base64Encoded: response.audioWavBase64) else {
                throw ServerError.invalidResponse("the audio is not valid base64")
            }
            try player.load(data: wav)
            output = SynthesisOutput(wav: wav, response: response, inputKey: key)
            lastTiming = Self.timingSummary(response)
            print("[synthesis] \(lastTiming ?? "")")
            serverNotices = response.warnings
            attemptedWithoutInput = false
            player.play()
        } catch {
            guard visitor == self.visitor else { return }
            if let error = error as? ServerError {
                serverNotices = error.notices.filter { $0.level != .error }
                if case .http(_, let message, _, _) = error { generationError = message } else { generationError = error.localizedDescription }
            } else {
                generationError = error.localizedDescription
            }
        }
    }

    /// Timing of a generation, for staff only (Settings > General and the app's standard output):
    /// "1.8 s of speech in 0.52 s, RTF 0.29, exact".
    static func timingSummary(_ response: SynthesisResponse) -> String {
        var summary = String(format: "%.2f s of speech in %.2f s", response.durationSeconds, response.elapsedSeconds)
        if let rtf = response.rtf { summary += String(format: ", RTF %.3f", rtf) }
        if let compute = response.compute { summary += ", \(compute)" }
        if response.fellBack == true { summary += " (fell back to float32)" }
        if response.dryRun == true { summary += ", dry run" }
        return summary
    }

    // -----------------------------
    // The primary action: one button in a fixed place, changing with the state
    // -----------------------------
    enum PrimaryAction: Equatable {
        case record, stopRecording, generate, generating, play, playAgain, stopPlayback, resetting

        var title: String {
            switch self {
            case .record: "Record your voice"
            case .stopRecording: "Stop"
            case .generate: "Generate my voice"
            case .generating: "Generating..."
            case .play: "Play"
            case .playAgain: "Play again"
            case .stopPlayback: "Stop"
            case .resetting: "Clearing..."
            }
        }

        var systemImage: String? {
            switch self {
            case .record: "mic.fill"
            case .stopRecording, .stopPlayback: "stop.fill"
            case .generate: "waveform"
            case .play, .playAgain: "play.fill"
            case .generating, .resetting: nil  // a progress indicator instead
            }
        }

        var shortcutHint: String {
            switch self {
            case .record, .stopRecording: "Space or \u{2318}R"
            case .generate: "Space, Return in the text or \u{2318}G"
            case .play, .playAgain, .stopPlayback: "Space or \u{2318}P"
            case .generating, .resetting: ""
            }
        }
    }

    /// Record, then stop, generate, play, and play again (Next visitor is a secondary action).
    var primaryAction: PrimaryAction {
        if isResetting { return .resetting }
        if recorder.isCapturing { return .stopRecording }
        if isSynthesizing { return .generating }
        if promptURL == nil { return .record }
        if outputIsCurrent {
            if player.isPlaying { return .stopPlayback }
            return player.timesFinished > 0 ? .playAgain : .play
        }
        return .generate
    }

    var primaryActionEnabled: Bool {
        switch primaryAction {
        case .record: canRecord
        case .generate: blockingReason == nil
        case .generating, .resetting: false
        case .stopRecording, .play, .playAgain, .stopPlayback: true
        }
    }

    func performPrimaryAction() {
        guard primaryActionEnabled else { return }
        switch primaryAction {
        case .record: startRecording()
        case .stopRecording: stopRecording()
        case .generate: generate()
        case .play, .playAgain, .stopPlayback: toggleOutputPlayback()
        case .generating, .resetting: break
        }
    }

    /// Return (or Command-Return) in the text field: generate, when that is the next step.
    func performPrimaryActionFromText() {
        if primaryAction == .generate, primaryActionEnabled {
            generate()
        }
    }

    /// Command-R: record, or stop the countdown or recording.
    func toggleRecording() {
        if recorder.isCapturing {
            stopRecording()
        } else if canRecord {
            startRecording()
        }
    }

    /// Command-P: play or stop the generated voice.
    func toggleOutputPlayback() {
        guard output != nil else { return }
        promptPlayer.stop()
        player.toggle()
    }

    // -----------------------------
    // Steps and instructions
    // -----------------------------
    /// 1 record, 2 type, 3 generate, 4 listen.
    var currentStep: Int {
        if recorder.isCapturing || promptURL == nil { return 1 }
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return 2 }
        return outputIsCurrent ? 4 : 3
    }

    func status(ofStep step: Int) -> StepStatus {
        let current = currentStep
        if step == current { return .current }
        if step < current { return .done }
        return step == 4 && output != nil ? .done : .upcoming
    }

    var instruction: Instruction {
        switch recorder.state {
        case .countdown:
            return Instruction(text: "Get ready to read the sentence aloud...", level: .info, systemImage: "timer")
        case .recording:
            return Instruction(text: "Read the sentence aloud now, then press Stop.", level: .info, systemImage: "waveform")
        case .idle:
            break
        }
        if isResetting {
            return Instruction(text: "Clearing everything for the next visitor...", level: .info, systemImage: "arrow.counterclockwise")
        }
        if readyForNextVisitor {
            return Instruction(text: "Ready for the next visitor.", level: .info, systemImage: "checkmark.circle")
        }
        if isSynthesizing {
            return Instruction(text: "Generating your voice. This takes a few seconds...", level: .info, systemImage: "waveform")
        }
        if let generationError {
            return Instruction(text: "Could not generate your voice: \(generationError) Please try again.",
                               level: .error, systemImage: "exclamationmark.triangle.fill")
        }
        switch currentStep {
        case 1:
            if selection == .recording, !canRecord, !isSynthesizing {
                return Instruction(text: "The microphone is not available. Use a sample voice in step 1.",
                                   level: .warning, systemImage: "mic.slash")
            }
            return Instruction(text: "Press \"Record your voice\" below, then read the sentence aloud.", level: .info, systemImage: "mic")
        case 2:
            return Instruction(text: "Type what you want your voice to say.", level: .info, systemImage: "keyboard")
        case 3:
            switch serverState {
            case .ready:
                return Instruction(text: "Press \"Generate my voice\" below. You can change the text first.", level: .info, systemImage: "sparkles")
            case .unknown, .loading:
                return Instruction(text: "The voice engine is starting. Please wait a moment.", level: .warning, systemImage: "hourglass")
            case .unreachable, .failed:
                return Instruction(text: "The voice engine is not running. Please ask a staff member.",
                                   level: .warning, systemImage: "exclamationmark.triangle.fill")
            }
        default:
            if player.isPlaying {
                return Instruction(text: "This is your voice.", level: .info, systemImage: "speaker.wave.2")
            }
            return Instruction(text: "Play it again, or press \"Next visitor\" when you're done.",
                               level: .info, systemImage: "checkmark.circle")
        }
    }

    // -----------------------------
    // Next visitor and quitting
    // -----------------------------
    /// Delete the recording, ask the server to forget it, and start over with the defaults.
    func nextVisitor() async {
        // Cancel what is in flight first (also its URLSession request), then reset: a late
        // response of the previous visitor is dropped by the visitor and prepare tags.
        synthesisTask?.cancel()
        progressTask?.cancel()
        visitor = UUID()
        dropPrepare()
        recorder.deleteAll()
        recorder.refreshAccess()
        player.unload()
        promptPlayer.unload()
        output = nil
        generationError = nil
        serverNotices = []
        dismissedNotices = []
        resetNotice = nil
        missingMicrophone = nil
        noMicrophone = false
        attemptedWithoutInput = false
        promptInfoCache = [:]
        selection = .recording
        text = config.demo.defaultText

        defer { showReadyForNextVisitor() }
        guard let client else { return }
        isResetting = true
        defer { isResetting = false }
        do {
            _ = try await client.reset()
        } catch let error as ServerError {
            if case .unreachable = error { return }  // not running: it keeps nothing (uploads are removed when it exits)
            resetNotice = Notice(level: .warning, code: "reset_failed", message:
                "The voice engine could not delete the last recording (\(error.localizedDescription)). Staff: restart it in Settings.")
        } catch {
            resetNotice = Notice(level: .warning, code: "reset_failed", message:
                "The voice engine could not delete the last recording (\(error.localizedDescription)).")
        }
    }

    /// Show "Ready for the next visitor" for a few seconds, or until the visitor starts.
    private func showReadyForNextVisitor() {
        readyForNextVisitor = true
        let visitor = self.visitor
        Task {
            try? await Task.sleep(for: .seconds(4))
            if self.visitor == visitor { readyForNextVisitor = false }
        }
    }

    /// While the app quits: delete the recordings, ask the server to forget the visitor (waiting
    /// at most 1.5 s) and stop a server the app started.
    func cleanUpForQuit() {
        prepareTask?.cancel()
        synthesisTask?.cancel()
        progressTask?.cancel()
        recorder.removeDirectory()
        if serverState == .ready, let client {
            client.resetBlocking(timeout: 1.5)
        }
        serverProcess.terminate()
    }

    // -----------------------------
    // Notices
    // -----------------------------
    /// The app's own checks, recomputed from the current state.
    var clientNotices: [Notice] {
        var notices: [Notice] = []
        if !consentAcknowledged {
            notices.append(Notice(level: .info, code: "consent", message:
                "Record only your own voice, and do not use the result to imitate anyone. "
                + "Your recording is deleted when you press \"Next visitor\"."))
        }
        if let resetNotice { notices.append(resetNotice) }
        switch serverState {
        case .unknown, .ready: break
        case .unreachable:
            notices.append(Notice(level: .warning, code: "server_unreachable", message:
                "The voice engine is not running. Staff: start it in Settings > General."))
        case .loading:
            notices.append(Notice(level: .info, code: "server_loading", message: "The voice engine is starting..."))
        case .failed:
            notices.append(Notice(level: .error, code: "server_failed", message:
                "The voice engine failed to start. Staff: see the server log in Settings > General."))
        }
        if health?.dryRun == true {
            notices.append(Notice(level: .info, code: "dry_run", message:
                "Test mode: the voice engine plays a tone instead of a voice."))
        }

        if selection == .recording {
            notices += microphoneNotices
        }

        let limits = config.warnings
        if attemptedWithoutInput && text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            notices.append(Notice(level: .warning, code: "text_empty", message: "Type something for your voice to say."))
        }
        if Double(text.count) > limits.maxTextChars {
            notices.append(Notice(level: .warning, code: "text_long", message:
                "That text is long (\(text.count) characters). Shorter sentences work best."))
        }
        if promptURL != nil {
            if transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                notices.append(Notice(level: .warning, code: "transcript_missing", message: selection == .recording
                    ? "There is no sentence to read. Staff: set one in Settings > Demo."
                    : "This sample voice has no transcript. Staff: add it in Settings > Presets."))
            }
            switch promptInfo {
            case .success(let info): notices += promptChecks(info)
            case .failure:
                notices.append(Notice(level: .error, code: "prompt_unreadable", message: selection == .recording
                    ? "The recording could not be read. Please record again."
                    : "This sample voice's audio file is missing. Staff: check Settings > Presets."))
            case nil: break
            }
        }
        return notices
    }

    private var microphoneNotices: [Notice] {
        var notices: [Notice] = []
        switch recorder.access {
        case .denied:
            notices.append(Notice(level: .error, code: "mic_denied", message:
                "Microphone access is off. Turn it on in System Settings > Privacy & Security > Microphone, "
                + "then press Record again. Or use a sample voice."))
        case .needsAppBundle:
            notices.append(Notice(level: .warning, code: "mic_needs_app", message:
                "Recording needs the app bundle: build it with demo/DeltaTTSDemo/make_app.sh. Sample voices still work."))
        case .granted, .undetermined:
            break
        }
        if let missingMicrophone {
            notices.append(Notice(level: .warning, code: "mic_missing", message:
                "\(missingMicrophone) is not connected; using the system default microphone."))
        }
        if let missingOutput {
            notices.append(Notice(level: .warning, code: "output_missing", message:
                "\(missingOutput) is not connected; using the system default output."))
        }
        if let outputNotice { notices.append(outputNotice) }
        if noMicrophone {
            notices.append(Notice(level: .error, code: "mic_none", message:
                "No microphone found. Connect one, or use a sample voice."))
        }
        if let failure = recorder.failure {
            notices.append(Notice(level: .error, code: "recording_failed", message:
                "The microphone could not record (\(failure)). Check the microphone, or use a sample voice."))
        }
        if let take = recorder.take, take.peakDecibels < -55 {
            notices.append(Notice(level: .warning, code: "recording_silent", message:
                "We could not hear you. Move closer to the microphone and record again."))
        }
        return notices
    }

    /// The extra button of a notice, if it has one.
    func action(for notice: Notice) -> NoticeAction? {
        switch notice.code {
        case "consent":
            return NoticeAction(title: "Don't Show Again") { self.consentAcknowledged = true }
        case "mic_denied":
            return NoticeAction(title: "Open System Settings") {
                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                    NSWorkspace.shared.open(url)
                }
            }
        default:
            return nil
        }
    }

    private func promptChecks(_ info: PromptInfo) -> [Notice] {
        let limits = config.warnings
        let mine = selection == .recording
        var notices: [Notice] = []
        let seconds = String(format: "%.1f", info.duration)
        if info.sampleRate < 16000 {
            notices.append(Notice(level: .error, code: "prompt_low_sample_rate", message:
                "The audio quality is too low (\(Int(info.sampleRate)) Hz; at least 16000 Hz is needed)."))
        } else if info.sampleRate < limits.minPromptSampleRate {
            notices.append(Notice(level: .warning, code: "prompt_low_sample_rate", message:
                "The audio quality is low (\(Int(info.sampleRate)) Hz); the voice may sound muffled."))
        }
        if info.duration < limits.minPromptSeconds {
            notices.append(Notice(level: .warning, code: "prompt_short", message: mine
                ? "Your recording is only \(seconds) s. Record again and read the whole sentence."
                : "This sample voice is only \(seconds) s long; the result may sound less like it."))
        } else if info.duration > limits.maxPromptSeconds {
            notices.append(Notice(level: info.duration > 30 && !config.processing.trimPrompt ? .error : .warning,
                                  code: "prompt_long", message:
                "The recording is \(seconds) s long; keep it under \(limits.maxPromptSeconds.formatted()) s."))
        }
        return notices
    }

    /// Notices about the recording itself, shown in step 1 next to the recording, so that a
    /// visitor can record again right away.
    nonisolated static let recordingCodes: Set<String> = [
        "prompt_short", "prompt_long", "prompt_low_sample_rate", "prompt_unreadable", "prompt_no_speech",
        "recording_silent", "audio_extra_missing",
    ]

    /// Step 1's notices: from /prepare, else from the last synthesis, else the app's own checks.
    var recordingNotices: [Notice] {
        var seen = Set<String>()
        return (prepareNotices + serverNotices + clientNotices)
            .filter { Self.recordingCodes.contains($0.code) && seen.insert($0.code).inserted }
    }

    /// Client checks and the server's notices of the last request, minus dismissed ones and the
    /// recording's notices (in step 1). A server notice is hidden when a client check reports the
    /// same thing (same code).
    var visibleNotices: [Notice] {
        let client = clientNotices.filter { !Self.recordingCodes.contains($0.code) }
        let codes = Set(client.map(\.code))
        let server = serverNotices.filter { !codes.contains($0.code) && !Self.recordingCodes.contains($0.code) }
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
            text = config.demo.defaultText
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
