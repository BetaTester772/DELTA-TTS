import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The Settings scene (Cmd-,), for staff: every section of the config, the voice engine (server)
/// and its log. Changes are saved right away.
struct SettingsView: View {
    var body: some View {
        TabView {
            Tab("General", systemImage: "gearshape") { GeneralSettings() }
            Tab("Demo", systemImage: "mic") { DemoSettings() }
            Tab("Server", systemImage: "server.rack") { ServerSettings() }
            Tab("Presets", systemImage: "person.wave.2") { PresetSettings() }
            Tab("Synthesis", systemImage: "slider.horizontal.3") { SynthesisSettings() }
            Tab("Processing", systemImage: "wand.and.stars") { ProcessingSettings() }
            Tab("Warnings", systemImage: "exclamationmark.triangle") { WarningSettings() }
        }
        .frame(width: 700, height: 600)
        .tint(Theme.accent)
        .preferredColorScheme(.light)
    }
}

// -----------------------------
// General: connection, repository, reset
// -----------------------------
struct GeneralSettings: View {
    @Environment(AppModel.self) private var model
    @State private var choosingRepository = false
    @State private var confirmingReset = false

    var body: some View {
        @Bindable var model = model
        Form {
            Section {
                LabeledContent("Status", value: engineStatus)
                LabeledContent("Last generation", value: model.lastTiming ?? "None yet")
                HStack {
                    if model.serverProcess.isRunning {
                        Button("Stop Server", systemImage: "stop.circle") { model.serverProcess.stop() }
                    } else {
                        Button("Start Server", systemImage: "play.circle") { model.startServer() }
                            .disabled(model.serverState == .ready || model.serverState == .loading)
                    }
                    Spacer()
                    Button("Copy Log") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.serverProcess.log, forType: .string)
                    }
                    Button("Clear Log") { model.serverProcess.clearLog() }
                }
                ScrollView {
                    Text(model.serverProcess.log.isEmpty ? "The app has not started a server." : model.serverProcess.log)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .defaultScrollAnchor(.bottom)
                .frame(height: 140)
            } header: {
                Text("Voice engine (server)")
            } footer: {
                Text("Start Server runs `uv run python demo/server.py --config <this app's config>` in the repository; it stops when the app quits. A server started in a terminal is used as well.")
            }

            Section {
                TextField("Server URL", text: $model.config.app.serverUrl,
                          prompt: Text(model.config.defaultServerURLString))
            } header: {
                Text("Connection")
            } footer: {
                Text("Leave empty to use the host and port of the Server tab.")
            }

            Section {
                LabeledContent("Repository") {
                    HStack {
                        TextField("Repository", text: $model.config.app.repoPath, prompt: Text("Path to the DELTA-TTS checkout"))
                            .labelsHidden()
                        Button("Choose...") { choosingRepository = true }
                    }
                }
                if ConfigStore.isRepository(model.config.app.repoPath) {
                    Label("demo/config.json found", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(Theme.accent)
                } else {
                    Label("No demo/config.json at this path", systemImage: "xmark.circle.fill")
                        .foregroundStyle(Theme.error)
                }
                TextField("uv", text: $model.config.app.uvPath, prompt: Text("Search Homebrew, ~/.local/bin and PATH"))
                Toggle(isOn: $model.config.app.launchDryRun) {
                    Text("Start the server in dry-run mode")
                    Text("No model is loaded; outputs are tones as long as the estimated speech.")
                }
            } header: {
                Text("Local server")
            }

            Section {
                LabeledContent("Text size") {
                    HStack {
                        Slider(value: $model.config.app.uiScale, in: 0.7...1.8, step: 0.05)
                        Text(String(format: "%.0f%%", model.config.app.uiScale * 100))
                            .monospacedDigit()
                            .frame(width: 52, alignment: .trailing)
                    }
                }
                Toggle(isOn: $model.config.app.showSaveButton) {
                    Text("Show a Save button for the generated voice")
                    Text("Saved files are not deleted by Next visitor.")
                }
            } header: {
                Text("Main screen")
            }

            Section {
                HStack {
                    Button("Reset to Defaults...") { confirmingReset = true }
                    Button("Show Config File") {
                        try? ConfigStore.save(model.config)
                        NSWorkspace.shared.activateFileViewerSelecting([ConfigStore.userConfigURL])
                    }
                }
            } footer: {
                Text("Settings are saved to \(ConfigStore.userConfigURL.path). Reset reloads demo/config.json from the repository.")
            }
        }
        .formStyle(.grouped)
        .fileImporter(isPresented: $choosingRepository, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result {
                model.config.app.repoPath = url.path
                model.presetsChanged()
            }
        }
        .confirmationDialog("Reset all settings to demo/config.json?", isPresented: $confirmingReset) {
            Button("Reset", role: .destructive) { model.resetToDefaults() }
        } message: {
            Text("Every setting is replaced except the repository and uv paths.")
        }
    }

    private var engineStatus: String {
        switch model.serverState {
        case .unknown: "Checking \(model.config.serverURL?.absoluteString ?? "")"
        case .unreachable(let reason): "\(model.config.serverURL?.absoluteString ?? "Invalid URL"): \(reason)"
        case .loading: "Loading the model"
        case .failed(let error): "Failed: \(error)"
        case .ready:
            [model.health?.dryRun == true ? "Ready (dry run)" : "Ready", model.health?.device, model.health?.compute]
                .compactMap { $0 }.joined(separator: " · ")
        }
    }
}

// -----------------------------
// Demo: what visitors read, say and record
// -----------------------------
struct DemoSettings: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        Form {
            Section {
                TextField("Title", text: $model.config.demo.title)
                TextField("Subtitle", text: $model.config.demo.subtitle, axis: .vertical)
                    .lineLimit(1...3)
                TextField("Venue", text: $model.config.demo.venue)
            } header: {
                Text("Header")
            } footer: {
                Text("Shown at the top of the main screen. Leave the subtitle or venue empty to hide it.")
            }

            Section {
                TextField("Sentence to read aloud", text: $model.config.demo.defaultTranscript, axis: .vertical)
                    .lineLimit(2...4)
                TextField("Text to say", text: $model.config.demo.defaultText, axis: .vertical)
                    .lineLimit(2...4)
            } header: {
                Text("Defaults")
            } footer: {
                Text("Visitors read the sentence aloud while recording; it is sent as the transcript of the recording. The text to say is prefilled, and both return on Next visitor.")
            }

            Section {
                LabeledContent("Stop recording after") {
                    Stepper(value: $model.config.recording.maxSeconds, in: 4...30, step: 1) {
                        Text("\(Int(model.config.recording.maxSeconds)) s").monospacedDigit()
                    }
                }
                LabeledContent("Countdown") {
                    Stepper(value: $model.config.recording.countdownSeconds, in: 0...5) {
                        Text("\(model.config.recording.countdownSeconds) s").monospacedDigit()
                    }
                }
                Picker("Sample rate", selection: $model.config.recording.sampleRate) {
                    ForEach([16000.0, 22050, 24000, 44100, 48000], id: \.self) { rate in
                        Text("\(Int(rate)) Hz").tag(rate)
                    }
                }
                Toggle(isOn: $model.config.recording.trimSilence) {
                    Text("Cut silence before and after the speech")
                    Text("Keeps the speaking-rate estimate right when visitors start late or stop late.")
                }
                Picker("Microphone", selection: microphone) {
                    Text("System default" + (model.audioDevices.defaultInputName.map { " (\($0))" } ?? "")).tag("")
                    ForEach(model.audioDevices.inputs) { device in
                        Text(device.name).tag(device.id)
                    }
                    if !model.config.recording.inputDeviceUid.isEmpty,
                       !model.audioDevices.inputs.contains(where: { $0.id == model.config.recording.inputDeviceUid }) {
                        Text(model.microphoneLabel).tag(model.config.recording.inputDeviceUid)
                    }
                }
                Picker("Output", selection: output) {
                    Text("System default" + (model.audioDevices.defaultOutputName.map { " (\($0))" } ?? "")).tag("")
                    ForEach(model.audioDevices.outputs) { device in
                        Text(device.name).tag(device.id)
                    }
                    if !model.config.app.outputDeviceUid.isEmpty,
                       !model.audioDevices.outputs.contains(where: { $0.id == model.config.app.outputDeviceUid }) {
                        Text("\(model.config.app.outputDeviceName) (not connected)").tag(model.config.app.outputDeviceUid)
                    }
                }
                LabeledContent("Microphone access", value: microphoneStatus)
                if model.recorder.access == .denied {
                    Button("Open Privacy & Security Settings") {
                        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                }
            } header: {
                Text("Recording")
            } footer: {
                Text("Mono 16-bit WAV. A chosen microphone is remembered; when it is not connected, the system default is used. A chosen output gives way to the system default when the default changes (e.g. headphones plugged in). Both are also on the main screen. Recordings stay in a private temporary folder and are deleted on Next visitor and when the app quits.")
            }
        }
        .formStyle(.grouped)
        .onAppear { model.recorder.refreshAccess() }
    }

    private var output: Binding<String> {
        Binding(get: { model.config.app.outputDeviceUid },
                set: { uid in model.chooseOutput(model.audioDevices.outputs.first { $0.id == uid }) })
    }

    private var microphone: Binding<String> {
        Binding(get: { model.config.recording.inputDeviceUid },
                set: { uid in model.chooseMicrophone(model.audioDevices.inputs.first { $0.id == uid }) })
    }

    private var microphoneStatus: String {
        switch model.recorder.access {
        case .granted: "Allowed"
        case .undetermined: "Asked on the first recording"
        case .denied: "Denied: allow DeltaTTSDemo in System Settings > Privacy & Security > Microphone"
        case .needsAppBundle: "Unavailable: run DeltaTTSDemo.app built by make_app.sh"
        }
    }
}

// -----------------------------
// Server: what demo/server.py loads
// -----------------------------
struct ServerSettings: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        Form {
            Section {
                TextField("Host", text: $model.config.server.host)
                TextField("Port", value: $model.config.server.port, format: .number.grouping(.never))
                Picker("Device", selection: device) {
                    Text("Automatic (CUDA, then MPS, then CPU)").tag("")
                    Text("MPS (Apple Silicon)").tag("mps")
                    Text("CUDA").tag("cuda")
                    Text("CPU").tag("cpu")
                }
            } header: {
                Text("Server")
            } footer: {
                Text("Read by the server when the app starts it; restart the server to apply. Keep the host at 127.0.0.1: the server is for local use only. The compute mode is in the Synthesis tab.")
            }

            Section("Models") {
                TextField("CosyVoice 3 directory", text: $model.config.server.backboneDir)
                TextField("DELTA-TTS checkpoint", text: $model.config.server.checkpoint)
                TextField("RE-USE directory or Hugging Face id", text: $model.config.server.reuseDir)
            }
        }
        .formStyle(.grouped)
    }

    private var device: Binding<String> {
        Binding(get: { model.config.server.device ?? "" },
                set: { model.config.server.device = $0.isEmpty ? nil : $0 })
    }
}

// -----------------------------
// Presets
// -----------------------------
struct PresetSettings: View {
    @Environment(AppModel.self) private var model
    @State private var selectedID: UUID?
    @State private var choosingWav = false

    var body: some View {
        @Bindable var model = model
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                List(selection: $selectedID) {
                    ForEach(model.config.presets) { preset in
                        Text(preset.name.isEmpty ? "Untitled" : preset.name).tag(preset.id)
                    }
                    .onMove { model.config.presets.move(fromOffsets: $0, toOffset: $1) }
                }
                HStack(spacing: 4) {
                    Button("Add", systemImage: "plus") { add() }
                    Button("Remove", systemImage: "minus") { remove() }
                        .disabled(selectedID == nil)
                    Spacer()
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
                .padding(8)
            }
            .frame(width: 200)

            Divider()

            if let preset = $model.config.presets.first(where: { $0.wrappedValue.id == selectedID }) {
                Form {
                    TextField("Name", text: preset.name)
                    LabeledContent("Audio") {
                        HStack {
                            TextField("Audio", text: preset.wav, prompt: Text("Relative to the repository, or absolute"))
                                .labelsHidden()
                            Button("Choose...") { choosingWav = true }
                        }
                    }
                    if FileManager.default.fileExists(atPath: model.config.resolve(preset.wrappedValue.wav).path) {
                        Label("File found", systemImage: "checkmark.circle.fill").foregroundStyle(Theme.accent)
                    } else {
                        Label("File not found", systemImage: "xmark.circle.fill").foregroundStyle(Theme.error)
                    }
                    TextField("Transcript", text: preset.transcript, axis: .vertical)
                        .lineLimit(2...5)
                }
                .formStyle(.grouped)
            } else {
                ContentUnavailableView("No preset selected", systemImage: "person.wave.2",
                                       description: Text("Presets are prompt audio files with their transcripts."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .fileImporter(isPresented: $choosingWav, allowedContentTypes: [.audio]) { result in
            guard case .success(let url) = result,
                  let index = model.config.presets.firstIndex(where: { $0.id == selectedID }) else { return }
            model.config.presets[index].wav = model.config.repoRelative(url)
        }
        .onAppear { selectedID = selectedID ?? model.config.presets.first?.id }
    }

    private func add() {
        let preset = DemoConfig.Preset(name: "New preset", wav: "", transcript: "")
        model.config.presets.append(preset)
        selectedID = preset.id
    }

    private func remove() {
        model.config.presets.removeAll { $0.id == selectedID }
        selectedID = model.config.presets.first?.id
        model.presetsChanged()
    }
}

// -----------------------------
// Synthesis
// -----------------------------
struct SynthesisSettings: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        Form {
            Section {
                NumberRow(title: "Length coefficient", value: $model.config.synthesis.lengthScale, range: 0.5...2.0, step: 0.05)
            } header: {
                Text("Length")
            } footer: {
                Text("Multiplies the output length estimated from the prompt's speaking rate: above 1 gives slower speech, below 1 faster.")
            }

            Section {
                LabeledContent("Steps") {
                    Stepper(value: $model.config.synthesis.numSteps, in: 1...64) {
                        TextField("Steps", value: $model.config.synthesis.numSteps, format: .number)
                            .labelsHidden()
                            .frame(width: 56)
                            .multilineTextAlignment(.trailing)
                    }
                }
                NumberRow(title: "Temperature", value: $model.config.synthesis.temperature, range: 0...1.5, step: 0.05)
                NumberRow(title: "Top-p", value: $model.config.synthesis.topP, range: 0.05...1.0, step: 0.05)
                NumberRow(title: "Time shift", value: $model.config.synthesis.tShift, range: 0.05...2.0, step: 0.05)
            } header: {
                Text("Decoding")
            } footer: {
                Text("Steps: unmasking iterations. Temperature 0 takes the most likely token (deterministic); above 0 samples. Top-p: nucleus filtering before the confidences. Time shift below 1 commits fewer tokens in early steps; 1 is linear.")
            }

            Section {
                Picker("Mode", selection: $model.computeMode) {
                    ForEach(ComputeMode.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
            } header: {
                Text("Compute mode")
            } footer: {
                Text("Exact runs DELTA-TTS in float32 and reproduces the paper. Fast runs it in float16: faster, with numerically different outputs (a step that overflows is redone in float32). Also on the main screen as Mode.")
            }
        }
        .formStyle(.grouped)
    }
}

/// A slider with a number field.
struct NumberRow: View {
    var title: String
    @Binding var value: Double
    var range: ClosedRange<Double>
    var step: Double

    var body: some View {
        LabeledContent(title) {
            HStack {
                Slider(value: $value, in: range, step: step)
                TextField(title, value: $value, format: .number.precision(.fractionLength(2)))
                    .labelsHidden()
                    .frame(width: 64)
                    .multilineTextAlignment(.trailing)
            }
        }
    }
}

// -----------------------------
// Processing
// -----------------------------
struct ProcessingSettings: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        Form {
            Section {
                Toggle(isOn: $model.config.processing.enhancePrompt) {
                    Text("Enhance the prompt")
                    Text("RE-USE removes noise, reverberation, clipping and codec artifacts before synthesis.")
                }
                Toggle(isOn: $model.config.processing.trimPrompt) {
                    Text("Trim the prompt")
                    Text("Silero VAD removes leading and trailing non-speech; the transcript stays valid.")
                }
                Toggle(isOn: $model.config.processing.trimOutput) {
                    Text("Trim the output")
                    Text("Silero VAD removes leading and trailing non-speech of the result.")
                }
            } footer: {
                Text("Needs the audio extra (uv sync --extra mac --extra audio). The first request with new settings loads the models, and prompt features are recomputed when the prompt processing changes. The paper results use none of these.")
            }
        }
        .formStyle(.grouped)
    }
}

// -----------------------------
// Warnings
// -----------------------------
struct WarningSettings: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        Form {
            Section("Prompt") {
                TextField("Minimum duration (s)", value: $model.config.warnings.minPromptSeconds, format: .number)
                TextField("Maximum duration (s)", value: $model.config.warnings.maxPromptSeconds, format: .number)
                TextField("Minimum sample rate (Hz)", value: $model.config.warnings.minPromptSampleRate,
                          format: .number.grouping(.never))
            }
            Section {
                TextField("Maximum text length (characters)", value: $model.config.warnings.maxTextChars,
                          format: .number.grouping(.never))
                TextField("Maximum estimated output (s)", value: $model.config.warnings.maxOutputSeconds, format: .number)
            } header: {
                Text("Text and output")
            } footer: {
                Text("The thresholds are sent with each request, so the server warns with the same limits.")
            }
            Section {
                Button("Show the Voice-Cloning Notice Again") { model.consentAcknowledged = false }
                    .disabled(!model.consentAcknowledged)
            }
        }
        .formStyle(.grouped)
    }
}
