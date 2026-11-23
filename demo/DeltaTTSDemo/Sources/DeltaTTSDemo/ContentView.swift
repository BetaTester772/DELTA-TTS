import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The main window, for visitors.
struct ContentView: View {
    @Environment(AppModel.self) private var model
    @State private var width: CGFloat = 1280
    @State private var exporting = false

    var body: some View {
        ScrollView {
            MainContent(width: width, exporting: $exporting)
        }
        .scrollBounceBehavior(.basedOnSize)
        .scrollIndicators(.never)  // a legacy scroller would narrow the content and could make the layout oscillate
        .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { width = $0 }
        .background(Theme.page)
        .mainScreenStyle(scale: model.config.app.uiScale)
        .frame(minWidth: 760, minHeight: 600)
        .animation(.smooth(duration: 0.35), value: model.recorder.isCapturing)
        .task { await model.pollHealth() }
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

extension View {
    /// Light appearance, near-black text and the main screen's scale. No tint: glass buttons
    /// would take it as their fill.
    func mainScreenStyle(scale: Double) -> some View {
        environment(\.uiScale, scale)
            .foregroundStyle(Theme.primaryText)
            .preferredColorScheme(.light)
    }
}

/// Header, what to do now, notices and the four steps: 1 read and record, 2 type what to say,
/// 3 generate, 4 listen. While recording, the sentence to read takes the whole width.
struct MainContent: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale
    var width: CGFloat
    @Binding var exporting: Bool

    static let maxWidth: CGFloat = 1600

    var body: some View {
        let padding = 32 * scale
        let gap = Spacing.gap * scale
        let contentWidth = min(width, Self.maxWidth) - 2 * padding
        VStack(alignment: .leading, spacing: gap) {
            HeaderView()
            InstructionBar()
            if model.recorder.isCapturing {
                RecordingFocusView(contentWidth: contentWidth)
                    .transition(.opacity)
            } else {
                NoticeStack()
                if contentWidth >= 900 * scale {
                    RecordStep()
                    HStack(alignment: .top, spacing: gap) {
                        TextStep()
                        GenerateStep()
                    }
                    ListenStep(compact: false, exporting: $exporting)
                } else {
                    RecordStep()
                    TextStep()
                    GenerateStep()
                    ListenStep(compact: true, exporting: $exporting)
                }
            }
        }
        .padding(.horizontal, padding)
        .padding(.top, 30)  // clear of the window buttons (hidden title bar)
        .padding(.bottom, 24 * scale)
        .frame(maxWidth: Self.maxWidth)
        .frame(maxWidth: .infinity)
    }
}

// -----------------------------
// Header, what to do now, notices
// -----------------------------
struct HeaderView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        let demo = model.config.demo
        VStack(alignment: .leading, spacing: 4 * scale) {
            HStack(alignment: .center, spacing: 20 * scale) {
                HStack(alignment: .firstTextBaseline, spacing: 14 * scale) {
                    Text(demo.title)
                        .scaledFont(TypeSize.title, weight: .semibold)
                        .tracking(-0.4)
                    if !demo.venue.isEmpty {
                        Text(demo.venue)
                            .scaledFont(TypeSize.venue, weight: .medium)
                            .foregroundStyle(Theme.tertiaryText)
                    }
                }
                Spacer(minLength: 16)
                EngineStatus()
                GlassEffectContainer(spacing: 8 * scale) {
                    HStack(spacing: 8 * scale) {
                        NextVisitorButton(prominent: false)
                        SettingsLink {
                            Image(systemName: "gearshape")
                                .font(.system(size: 16 * scale))
                                .padding(4 * scale)
                        }
                        .buttonStyle(.glass)
                        .controlSize(.large)
                        .help("Settings (for staff)")
                    }
                }
            }
            if !demo.subtitle.isEmpty {
                Text(demo.subtitle)
                    .scaledFont(TypeSize.subtitle)
                    .foregroundStyle(Theme.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Rectangle()
                .fill(Theme.hairline)
                .frame(height: 1)
                .padding(.top, 12 * scale)
        }
    }
}

/// Small status of the server, in plain words.
struct EngineStatus: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        let status = self.status
        HStack(spacing: 8) {
            Circle()
                .fill(status.color)
                .frame(width: 8 * scale, height: 8 * scale)
            Text(status.label)
                .scaledFont(TypeSize.caption)
                .foregroundStyle(Theme.secondaryText)
                .lineLimit(1)
        }
        .help(model.config.serverURL?.absoluteString ?? "Invalid server URL")
    }

    private var status: (label: String, color: Color) {
        switch model.serverState {
        case .unknown: return ("Checking...", Theme.idle)
        case .unreachable: return ("Voice engine offline", Theme.idle)
        case .loading: return ("Voice engine starting", Theme.warning)
        case .failed: return ("Voice engine error", Theme.error)
        case .ready: return (model.health?.dryRun == true ? "Test mode" : "Voice engine ready", Theme.accent)
        }
    }
}

/// What to do now, in one line.
struct InstructionBar: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        let instruction = model.instruction
        HStack(spacing: 12 * scale) {
            RoundedRectangle(cornerRadius: 2)
                .fill(instruction.level.color)
                .frame(width: 3 * scale)
            Text(instruction.text)
                .scaledFont(TypeSize.instruction, weight: .medium)
                .foregroundStyle(instruction.level == .info ? Theme.primaryText : instruction.level.color)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
            if model.isSynthesizing || model.isResetting {
                ProgressView().controlSize(.small)
            }
        }
        .padding(.vertical, 2 * scale)
        .fixedSize(horizontal: false, vertical: true)
        .animation(.smooth, value: instruction)
    }
}

struct NoticeStack: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        let notices = model.visibleNotices
        if !notices.isEmpty {
            VStack(spacing: 8 * scale) {
                ForEach(notices) { notice in
                    NoticeBanner(notice: notice,
                                 action: model.action(for: notice),
                                 dismiss: { withAnimation(.snappy) { model.dismiss(notice) } })
                        .transition(.opacity)
                }
            }
            .animation(.snappy, value: notices.map(\.id))
        }
    }
}

/// One dismissible notice on a muted tint of its level.
struct NoticeBanner: View {
    @Environment(\.uiScale) private var scale
    var notice: Notice
    var action: NoticeAction?
    var dismiss: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 10 * scale) {
            Image(systemName: notice.level.systemImage)
                .foregroundStyle(notice.level.color)
            Text(notice.message)
                .foregroundStyle(Theme.primaryText)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
            if let action {
                Button(action.title, action: action.perform)
                    .buttonStyle(.borderless)
                    .foregroundStyle(Theme.accent)
                    .fontWeight(.semibold)
            }
            Button(action: dismiss) {
                Image(systemName: "xmark")
                    .font(.system(size: 12 * scale, weight: .semibold))
                    .foregroundStyle(Theme.tertiaryText)
                    .padding(6 * scale)
                    .contentShape(.rect)
            }
            .buttonStyle(.borderless)
            .help("Dismiss")
        }
        .scaledFont(TypeSize.banner)
        .padding(.horizontal, 14 * scale)
        .padding(.vertical, 8 * scale)
        .background(notice.level.tint, in: .rect(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10).strokeBorder(notice.level.color.opacity(0.22), lineWidth: 1)
        }
    }
}

// -----------------------------
// 1 Read and record
// -----------------------------
struct RecordStep: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        StepCard(number: 1, title: "Read and record", status: model.status(ofStep: 1)) {
            switch model.selection {
            case .recording: ownVoice
            case .preset: sampleVoice
            }
        }
    }

    @ViewBuilder private var ownVoice: some View {
        Teleprompter(text: model.transcript, size: TypeSize.prompter * scale, active: false)
        HStack(spacing: 10 * scale) {
            Button(action: model.startRecording) {
                ActionLabel(title: model.recorder.take == nil ? "Record your voice" : "Record again", systemImage: "mic.fill")
            }
            .actionStyle(prominent: model.currentStep == 1)
            .controlSize(.large)
            .keyboardShortcut("r", modifiers: .command)
            .disabled(!model.canRecord)
            if let take = model.recorder.take {
                Button(action: model.togglePromptPlayback) {
                    ActionLabel(title: model.promptPlayer.isPlaying ? "Stop" : "Play",
                                systemImage: model.promptPlayer.isPlaying ? "stop.fill" : "play.fill")
                }
                .buttonStyle(.glass)
                .controlSize(.large)
                Text("\(formatSeconds(take.duration)) recorded")
                    .scaledFont(TypeSize.caption)
                    .foregroundStyle(Theme.secondaryText)
            }
            Spacer(minLength: 12 * scale)
            MicrophoneMenu()
        }
        HStack(alignment: .firstTextBaseline, spacing: 12 * scale) {
            Text("Your recording is used only for this demo and is deleted on \u{201C}Next visitor\u{201D} or when the app quits.")
                .scaledFont(TypeSize.caption)
                .foregroundStyle(Theme.tertiaryText)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 12 * scale)
            SampleVoiceMenu(title: "No microphone? Use a sample voice")
        }
    }

    @ViewBuilder private var sampleVoice: some View {
        Text("Sample voice: \(model.selectedPreset?.name ?? "")")
            .scaledFont(TypeSize.body, weight: .semibold)
        Text("\u{201C}\(model.transcript)\u{201D}")
            .scaledFont(TypeSize.body)
            .foregroundStyle(Theme.secondaryText)
            .fixedSize(horizontal: false, vertical: true)
        HStack(spacing: 10 * scale) {
            Button(action: model.togglePromptPlayback) {
                ActionLabel(title: model.promptPlayer.isPlaying ? "Stop" : "Play sample",
                            systemImage: model.promptPlayer.isPlaying ? "stop.fill" : "play.fill")
            }
            .buttonStyle(.glass)
            .controlSize(.large)
            Button(action: model.useOwnVoice) {
                ActionLabel(title: "Use my own voice", systemImage: "mic.fill")
            }
            .buttonStyle(.glass)
            .controlSize(.large)
        }
        SampleVoiceMenu(title: "Choose another sample voice")
    }
}

/// The microphone to record from: the system default (followed when it changes) or a chosen one.
struct MicrophoneMenu: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Menu {
            Button("System default" + (model.audioDevices.defaultInputName.map { " (\($0))" } ?? "")) {
                model.chooseMicrophone(nil)
            }
            if !model.audioDevices.inputs.isEmpty {
                Divider()
                ForEach(model.audioDevices.inputs) { device in
                    Button(device.name) { model.chooseMicrophone(device) }
                }
            }
        } label: {
            Text("Microphone: \(model.microphoneLabel)")
                .scaledFont(TypeSize.caption, weight: .medium)
                .lineLimit(1)
        }
        .menuStyle(.button)
        .buttonStyle(.borderless)
        .fixedSize()
        .foregroundStyle(Theme.secondaryText)
        .disabled(model.recorder.isCapturing)
        .help("Staff: choose the microphone. It is also in Settings > Demo.")
    }
}

/// The fallback when the microphone fails: a compact menu of the presets.
struct SampleVoiceMenu: View {
    @Environment(AppModel.self) private var model
    var title: String

    var body: some View {
        Menu {
            ForEach(model.config.presets) { preset in
                Button(preset.name) { model.useSampleVoice(preset.id) }
            }
        } label: {
            Text(title)
                .scaledFont(TypeSize.caption, weight: .medium)
        }
        .menuStyle(.button)
        .buttonStyle(.borderless)
        .fixedSize()
        .foregroundStyle(Theme.accent)
        .disabled(model.config.presets.isEmpty)
    }
}

/// Countdown and recording: the sentence fills the width, with the level, the time and Stop.
struct RecordingFocusView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale
    var contentWidth: CGFloat

    var body: some View {
        let recorder = model.recorder
        // 46 points at ui_scale 1, up to 30% larger when the window is wider than 1350 points
        let prompterSize = TypeSize.prompterFocus * scale * min(1.3, max(1, contentWidth / 1350))
        VStack(spacing: 22 * scale) {
            switch recorder.state {
            case .countdown(let remaining):
                HStack(alignment: .firstTextBaseline, spacing: 14 * scale) {
                    Text("Get ready to read")
                        .scaledFont(TypeSize.stepTitle + 3, weight: .semibold)
                        .foregroundStyle(Theme.secondaryText)
                    Text("\(remaining)")
                        .font(.system(size: 56 * scale, weight: .bold).monospacedDigit())
                        .foregroundStyle(Theme.accent)
                        .contentTransition(.numericText(countsDown: true))
                }
            default:
                HStack(spacing: 10 * scale) {
                    Image(systemName: "circle.fill")
                        .font(.system(size: 12 * scale))
                        .foregroundStyle(Theme.error)
                        .symbolEffect(.pulse)
                    Text("Recording. Read aloud now.")
                        .scaledFont(TypeSize.stepTitle + 3, weight: .semibold)
                }
            }

            Teleprompter(text: model.transcript, size: prompterSize, active: true)

            VStack(spacing: 10 * scale) {
                LevelMeter(level: recorder.state == .recording ? recorder.level : 0)
                    .frame(height: 14 * scale)
                    .frame(maxWidth: 640 * scale)
                Text(recorder.state == .recording
                     ? "\(Int(recorder.elapsed.rounded(.down))) s of \(Int(recorder.maxSeconds)) s"
                     : " ")
                    .scaledFont(TypeSize.body)
                    .monospacedDigit()
                    .foregroundStyle(Theme.secondaryText)
            }

            Button(action: model.stopRecording) {
                ActionLabel(title: recorder.state == .recording ? "Stop" : "Cancel", systemImage: "stop.fill")
                    .frame(minWidth: 140 * scale)
            }
            .buttonStyle(.glassProminent)
            .tint(Theme.accent)
            .controlSize(.large)
            .keyboardShortcut("r", modifiers: .command)

            Text("Press Stop when you have finished the sentence. Recording stops by itself after \(Int(recorder.maxSeconds)) s.")
                .scaledFont(TypeSize.caption)
                .foregroundStyle(Theme.tertiaryText)
                .multilineTextAlignment(.center)
        }
        .padding(.horizontal, 28 * scale)
        .padding(.vertical, 30 * scale)
        .frame(maxWidth: .infinity)
        .card(highlighted: true)
    }
}

// -----------------------------
// 2 Type what to say
// -----------------------------
struct TextStep: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        @Bindable var model = model
        let limit = Int(model.config.warnings.maxTextChars)
        StepCard(number: 2, title: "Type what to say", status: model.status(ofStep: 2), minHeight: 168 * scale) {
            TextEditor(text: $model.text)
                .scaledFont(TypeSize.body)
                .foregroundStyle(Theme.primaryText)
                .scrollContentBackground(.hidden)
                .padding(8 * scale)
                .frame(height: 62 * scale)
                .background(Theme.well, in: .rect(cornerRadius: 10))
            HStack {
                Text("Change the words if you like.")
                Spacer()
                Text("\(model.text.count) / \(limit)")
                    .monospacedDigit()
                    .foregroundStyle(model.text.count > limit ? Theme.warning : Theme.tertiaryText)
            }
            .scaledFont(TypeSize.caption)
            .foregroundStyle(Theme.tertiaryText)
        }
    }
}

// -----------------------------
// 3 Generate
// -----------------------------
struct GenerateStep: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        StepCard(number: 3, title: "Generate", status: model.status(ofStep: 3), minHeight: 168 * scale) {
            HStack(spacing: 14 * scale) {
                Button(action: generate) {
                    HStack(spacing: 8 * scale) {
                        if model.isSynthesizing {
                            ProgressView().controlSize(.small)
                        }
                        Text(model.isSynthesizing ? "Generating..." : "Generate my voice")
                    }
                    .scaledFont(TypeSize.action, weight: .semibold)
                    .padding(.horizontal, 6 * scale)
                    .padding(.vertical, 4 * scale)
                }
                .actionStyle(prominent: model.currentStep == 3)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(model.isSynthesizing || model.blockingReason != nil)
                Text(hint)
                    .scaledFont(TypeSize.caption)
                    .foregroundStyle(Theme.tertiaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var hint: String {
        if model.isSynthesizing { return "This takes a few seconds." }
        return model.blockingReason ?? "Makes your voice say the text above."
    }

    private func generate() {
        Task { await model.synthesize() }
    }
}

// -----------------------------
// 4 Listen
// -----------------------------
struct ListenStep: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale
    var compact: Bool  // stacked, for narrow windows
    @Binding var exporting: Bool

    var body: some View {
        StepCard(number: 4, title: "Listen", status: model.status(ofStep: 4)) {
            if let output = model.output {
                let layout = compact
                    ? AnyLayout(VStackLayout(alignment: .leading, spacing: Spacing.item * scale))
                    : AnyLayout(HStackLayout(alignment: .center, spacing: 18 * scale))
                layout {
                    Button {
                        model.player.toggle()
                    } label: {
                        ActionLabel(title: model.player.isPlaying ? "Stop" : "Play again",
                                    systemImage: model.player.isPlaying ? "stop.fill" : "play.fill")
                    }
                    .buttonStyle(.glass)
                    .controlSize(.large)
                    progress
                        .frame(maxWidth: compact ? .infinity : 340 * scale)
                    VStack(alignment: .leading, spacing: 2 * scale) {
                        Text(model.speedSummary(output.response))
                            .foregroundStyle(Theme.secondaryText)
                        Text(model.outputLabel)
                            .foregroundStyle(Theme.tertiaryText)
                    }
                    .scaledFont(TypeSize.caption)
                    .fixedSize(horizontal: false, vertical: true)
                    if !compact { Spacer(minLength: 0) }
                    if model.config.app.showSaveButton {
                        Button("Save...") { exporting = true }
                            .buttonStyle(.glass)
                            .controlSize(.large)
                    }
                    if model.currentStep == 4 {
                        NextVisitorButton(prominent: true)
                    }
                }
            } else {
                HStack {
                    Text("Your generated voice will play here.")
                        .scaledFont(TypeSize.body)
                        .foregroundStyle(Theme.tertiaryText)
                    Spacer()
                    Text(model.outputLabel)
                        .scaledFont(TypeSize.caption)
                        .foregroundStyle(Theme.tertiaryText)
                }
            }
        }
    }

    private var progress: some View {
        TimelineView(.animation(minimumInterval: 0.05, paused: !model.player.isPlaying)) { _ in
            let duration = max(model.player.duration, 0.001)
            let position = min(model.player.currentTime, duration)
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: position, total: duration)
                    .tint(Theme.accent)
                HStack {
                    Text(formatSeconds(position))
                    Spacer()
                    Text(formatSeconds(model.player.duration))
                }
                .scaledFont(TypeSize.caption)
                .monospacedDigit()
                .foregroundStyle(Theme.tertiaryText)
            }
        }
    }
}

/// Deletes this visitor's recording and starts over (also File > Next visitor, Command-N).
/// Secondary in the header; the emphasized last action once the generated voice is ready.
struct NextVisitorButton: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale
    var prominent: Bool

    var body: some View {
        let button = Button {
            Task { await model.nextVisitor() }
        } label: {
            Label("Next visitor", systemImage: "arrow.counterclockwise")
                .scaledFont(prominent ? TypeSize.action : TypeSize.body, weight: .semibold)
                .padding(.horizontal, 4 * scale)
                .padding(.vertical, prominent ? 4 * scale : 0)
        }
        .controlSize(.large)
        .disabled(model.isResetting || model.recorder.isCapturing)
        .help("Delete this visitor's recording and start over (\u{2318}N)")
        if prominent {
            button.buttonStyle(.glassProminent).tint(Theme.accent)
        } else {
            button.buttonStyle(.bordered).tint(Theme.accent)
        }
    }
}
