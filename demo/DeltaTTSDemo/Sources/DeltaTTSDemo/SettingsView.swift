import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The Settings scene (Cmd-,): every section of the config. Changes are saved right away.
struct SettingsView: View {
    var body: some View {
        TabView {
            Tab("General", systemImage: "gearshape") { GeneralSettings() }
            Tab("Server", systemImage: "server.rack") { ServerSettings() }
            Tab("Presets", systemImage: "person.wave.2") { PresetSettings() }
            Tab("Synthesis", systemImage: "slider.horizontal.3") { SynthesisSettings() }
            Tab("Processing", systemImage: "wand.and.stars") { ProcessingSettings() }
            Tab("Warnings", systemImage: "exclamationmark.triangle") { WarningSettings() }
        }
        .frame(width: 640, height: 540)
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
                        .foregroundStyle(.green)
                } else {
                    Label("No demo/config.json at this path", systemImage: "xmark.circle.fill")
                        .foregroundStyle(.red)
                }
                TextField("uv", text: $model.config.app.uvPath, prompt: Text("Search Homebrew, ~/.local/bin and PATH"))
                Toggle(isOn: $model.config.app.launchDryRun) {
                    Text("Start the server in dry-run mode")
                    Text("No model is loaded; outputs are tones as long as the estimated speech.")
                }
            } header: {
                Text("Local server")
            } footer: {
                Text("Start Server runs `uv run python demo/server.py --config <this app's config>` in the repository.")
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
            Text("Server, presets, synthesis, processing and warning settings are replaced. The repository and uv paths are kept.")
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
                Picker("Compute at startup", selection: $model.config.server.compute) {
                    ForEach(ComputeMode.allCases) { Text($0.title).tag($0) }
                }
            } header: {
                Text("Server")
            } footer: {
                Text("Read by the server when the app starts it; restart the server to apply. Keep the host at 127.0.0.1: the server is for local use only.")
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
                        Label("File found", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                    } else {
                        Label("File not found", systemImage: "xmark.circle.fill").foregroundStyle(.red)
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
                Picker("Compute", selection: $model.config.synthesis.compute) {
                    ForEach(ComputeMode.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
            } header: {
                Text("Compute mode")
            } footer: {
                Text("Exact runs DELTA-TTS in float32. Fast runs it in float16: its outputs differ numerically from exact ones, and a step that overflows float16 is recomputed in float32, after which the rest of that utterance stays in float32. The first fast request creates a float16 copy of the model (about 1 GB).")
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
