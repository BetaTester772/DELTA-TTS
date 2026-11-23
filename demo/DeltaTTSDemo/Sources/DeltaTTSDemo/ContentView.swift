import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The main window: notices, prompt, text, synthesis controls and the output.
struct ContentView: View {
    @Environment(AppModel.self) private var model
    @State private var importingPrompt = false
    @State private var exporting = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HeaderView()
                NoticeStack()
                PromptCard(importing: $importingPrompt)
                TextCard()
                SynthesizeBar()
                if let output = model.output {
                    OutputCard(output: output, exporting: $exporting)
                }
            }
            .padding(20)
            .frame(maxWidth: 860)
            .frame(maxWidth: .infinity)
        }
        .background { Backdrop() }
        .frame(minWidth: 640, minHeight: 620)
        .task { await model.pollHealth() }
        .fileImporter(isPresented: $importingPrompt, allowedContentTypes: [.audio]) { result in
            switch result {
            case .success(let url): model.useCustomPrompt(url)
            case .failure(let error): model.alert = AlertItem(title: "Cannot open the file", message: error.localizedDescription)
            }
        }
        .fileExporter(isPresented: $exporting, document: model.output.map { WAVDocument(data: $0.wav) },
                      contentType: .wav, defaultFilename: "delta_tts.wav") { result in
            if case .failure(let error) = result {
                model.alert = AlertItem(title: "Cannot save the audio", message: error.localizedDescription)
            }
        }
        .alert(model.alert?.title ?? "", isPresented: alertShown, presenting: model.alert) { _ in
            Button("OK", role: .cancel) {}
        } message: { item in
            Text(item.message)
        }
    }

    private var alertShown: Binding<Bool> {
        Binding(get: { model.alert != nil }, set: { if !$0 { model.alert = nil } })
    }
}

// -----------------------------
// Header and server
// -----------------------------
struct HeaderView: View {
    @Environment(AppModel.self) private var model
    @State private var showingLog = false

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("DELTA-TTS")
                    .font(.largeTitle.bold())
                Text("Zero-shot speech synthesis in the voice of a short prompt")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            GlassEffectContainer(spacing: 8) {
                HStack(spacing: 8) {
                    ServerStatusPill()
                    if model.serverProcess.isRunning {
                        Button("Stop Server", systemImage: "stop.circle") { model.serverProcess.stop() }
                            .buttonStyle(.glass)
                    } else if case .unreachable = model.serverState {
                        Button("Start Server", systemImage: "play.circle") { model.startServer() }
                            .buttonStyle(.glass)
                            .help(model.config.app.launchDryRun ? "Starts the server in dry-run mode" : "Starts the server and loads the model")
                    }
                    Button { showingLog = true } label: {
                        Image(systemName: "doc.plaintext")
                    }
                    .buttonStyle(.glass)
                    .help("Server log")
                    .popover(isPresented: $showingLog) { ServerLogView() }
                    SettingsLink {
                        Image(systemName: "gearshape")
                    }
                    .buttonStyle(.glass)
                    .help("Settings")
                }
            }
        }
    }
}

struct ServerStatusPill: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let status = self.status
        let color = status.color
        HStack(spacing: 8) {
            if model.serverState == .loading {
                ProgressView().controlSize(.mini)
            } else {
                Circle().fill(color).frame(width: 8, height: 8)
            }
            Text(status.label)
                .font(.callout)
                .lineLimit(1)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .glassEffect(.regular.tint(color.opacity(0.25)), in: .capsule)
        .help(model.config.serverURL?.absoluteString ?? "Invalid server URL")
    }

    private var status: (label: String, color: Color) {
        switch model.serverState {
        case .unknown: return ("Checking...", Color.gray)
        case .unreachable: return ("Offline", Color.gray)
        case .loading: return ("Loading model", Color.orange)
        case .failed: return ("Failed", Color.red)
        case .ready:
            if model.health?.dryRun == true { return ("Dry run", Color.purple) }
            let parts = ["Ready", model.health?.device, model.health?.compute].compactMap { $0 }
            return (parts.joined(separator: " · "), Color.green)
        }
    }
}

struct ServerLogView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Server log").font(.headline)
                Spacer()
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(model.serverProcess.log, forType: .string)
                }
                Button("Clear") { model.serverProcess.clearLog() }
            }
            ScrollView {
                Text(model.serverProcess.log.isEmpty ? "The app has not started a server." : model.serverProcess.log)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .defaultScrollAnchor(.bottom)
        }
        .padding(14)
        .frame(width: 620, height: 360)
    }
}

// -----------------------------
// Notices
// -----------------------------
struct NoticeStack: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let notices = model.visibleNotices
        if !notices.isEmpty {
            GlassEffectContainer(spacing: 8) {
                VStack(spacing: 8) {
                    ForEach(notices) { notice in
                        NoticeBanner(notice: notice,
                                     dismiss: { withAnimation(.snappy) { model.dismiss(notice) } },
                                     dontShowAgain: dontShowAgainAction(for: notice))
                            .transition(.move(edge: .top).combined(with: .opacity))
                    }
                }
            }
            .animation(.snappy, value: notices.map(\.id))
        }
    }

    private func dontShowAgainAction(for notice: Notice) -> (() -> Void)? {
        guard notice.code == "consent" else { return nil }
        return { withAnimation(.snappy) { model.consentAcknowledged = true } }
    }
}

/// One dismissible notice, tinted by its level.
struct NoticeBanner: View {
    var notice: Notice
    var dismiss: () -> Void
    var dontShowAgain: (() -> Void)?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: notice.level.systemImage)
                .foregroundStyle(notice.level.color)
            Text(notice.message)
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
            if let dontShowAgain {
                Button("Don't Show Again", action: dontShowAgain)
                    .buttonStyle(.glass)
                    .controlSize(.small)
            }
            Button(action: dismiss) {
                Image(systemName: "xmark")
            }
            .buttonStyle(.glass)
            .buttonBorderShape(.circle)
            .controlSize(.small)
            .help("Dismiss")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .glassEffect(.regular.tint(notice.level.color.opacity(0.22)), in: .rect(cornerRadius: 16))
    }
}

// -----------------------------
// Inputs
// -----------------------------
struct PromptCard: View {
    @Environment(AppModel.self) private var model
    @Binding var importing: Bool

    var body: some View {
        @Bindable var model = model
        GlassCard(title: "Voice prompt", systemImage: "person.wave.2") {
            HStack(spacing: 10) {
                Picker("Prompt", selection: selection) {
                    if model.selection == .none {
                        Text("Choose a prompt").tag(PromptSelection.none)
                    }
                    ForEach(model.config.presets) { preset in
                        Text(preset.name).tag(PromptSelection.preset(preset.id))
                    }
                    if let url = model.customPromptURL {
                        Divider()
                        Text(url.lastPathComponent).tag(PromptSelection.file)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: 300)
                Button("Choose File...", systemImage: "folder") { importing = true }
                    .buttonStyle(.glass)
                Spacer()
                Button {
                    model.togglePromptPreview()
                } label: {
                    Label(model.promptPlayer.isPlaying ? "Stop" : "Listen",
                          systemImage: model.promptPlayer.isPlaying ? "stop.fill" : "play.fill")
                }
                .buttonStyle(.glass)
                .disabled(model.promptURL == nil)
            }
            promptInfo
            TextField("Transcript: the words spoken in the prompt", text: $model.transcript, axis: .vertical)
                .lineLimit(1...4)
                .textFieldStyle(.roundedBorder)
        }
    }

    private var selection: Binding<PromptSelection> {
        Binding(get: { model.selection }, set: { model.select($0) })
    }

    @ViewBuilder private var promptInfo: some View {
        switch model.promptInfo {
        case .success(let info)?:
            Text("\(formatSeconds(info.duration)) · \(Int(info.sampleRate)) Hz · \(model.promptURL?.lastPathComponent ?? "")")
                .font(.caption)
                .foregroundStyle(.secondary)
        case .failure(let error)?:
            Text(error.localizedDescription)
                .font(.caption)
                .foregroundStyle(.red)
        case nil:
            EmptyView()
        }
    }
}

struct TextCard: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        let limit = Int(model.config.warnings.maxTextChars)
        GlassCard(title: "Text", systemImage: "text.quote") {
            TextEditor(text: $model.text)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(8)
                .frame(minHeight: 140)
                .background(Color(nsColor: .textBackgroundColor).opacity(0.55), in: RoundedRectangle(cornerRadius: 12))
            HStack {
                Spacer()
                Text("\(model.text.count) / \(limit) characters")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(model.text.count > limit ? Color.orange : Color.secondary)
            }
        }
    }
}

struct SynthesizeBar: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: 6) {
            GlassEffectContainer(spacing: 12) {
                HStack(spacing: 12) {
                    Button(action: synthesize) {
                        HStack(spacing: 8) {
                            if model.isSynthesizing {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "waveform")
                            }
                            Text(model.isSynthesizing ? "Synthesizing..." : "Synthesize")
                        }
                        .frame(minWidth: 150)
                    }
                    .buttonStyle(.glassProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(model.isSynthesizing)

                    HStack(spacing: 8) {
                        Text("Length")
                        Slider(value: $model.config.synthesis.lengthScale, in: 0.5...2.0, step: 0.05)
                            .frame(width: 130)
                        Text(String(format: "×%.2f", model.config.synthesis.lengthScale))
                            .font(.callout.monospacedDigit())
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .glassEffect(.regular, in: .capsule)
                    .help("Length coefficient: multiplies the estimated output length (above 1: slower speech)")

                    Picker("Compute", selection: $model.config.synthesis.compute) {
                        Text("Exact").tag(ComputeMode.exact)
                        Text("Fast").tag(ComputeMode.fast)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(width: 140)
                    .help("Exact: float32. Fast: float16 with a float32 fallback on overflow; outputs differ numerically.")
                }
            }
            if !model.isSynthesizing, let reason = model.blockingReason {
                Text(reason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func synthesize() {
        Task { await model.synthesize() }
    }
}

// -----------------------------
// Output
// -----------------------------
struct OutputCard: View {
    @Environment(AppModel.self) private var model
    var output: SynthesisOutput
    @Binding var exporting: Bool

    var body: some View {
        let response = output.response
        GlassCard(title: "Output", systemImage: "speaker.wave.2") {
            HStack(spacing: 14) {
                Button {
                    model.player.toggle()
                } label: {
                    Image(systemName: model.player.isPlaying ? "stop.fill" : "play.fill")
                        .font(.title3)
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.glassProminent)
                .buttonBorderShape(.circle)
                .help(model.player.isPlaying ? "Stop" : "Play")

                TimelineView(.animation(minimumInterval: 0.05, paused: !model.player.isPlaying)) { _ in
                    let duration = max(model.player.duration, 0.001)
                    let position = min(model.player.currentTime, duration)
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: position, total: duration)
                        HStack {
                            Text(formatSeconds(position))
                            Spacer()
                            Text(formatSeconds(model.player.duration))
                        }
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    }
                }

                Button("Save...", systemImage: "square.and.arrow.down") { exporting = true }
                    .buttonStyle(.glass)
            }
            HStack(spacing: 22) {
                Stat(title: "Duration", value: formatSeconds(response.durationSeconds))
                Stat(title: "Elapsed", value: formatSeconds(response.elapsedSeconds))
                Stat(title: "RTF", value: response.rtf.map { String(format: "%.3f", $0) } ?? "-")
                Stat(title: "Compute", value: computeLabel(response))
                if response.dryRun == true {
                    Stat(title: "Mode", value: "Dry run")
                }
            }
            Text(output.text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(3)
        }
    }

    private func computeLabel(_ response: SynthesisResponse) -> String {
        let mode = response.compute ?? "-"
        return response.fellBack == true ? "\(mode) (fell back to float32)" : mode
    }
}

struct Stat: View {
    var title: String
    var value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.body.monospacedDigit())
        }
    }
}
