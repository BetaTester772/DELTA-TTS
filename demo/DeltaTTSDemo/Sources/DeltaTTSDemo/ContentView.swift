import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The main window, for visitors: the steps as a guide, and below them one fixed action bar whose
/// primary button does the next thing (record, stop, generate, play, next visitor), so the cursor
/// can stay in one place.
struct ContentView: View {
    @Environment(AppModel.self) private var model
    @State private var width: CGFloat = 1280
    @State private var exporting = false
    @FocusState private var textFocused: Bool

    var body: some View {
        ScrollView {
            MainContent(width: width, textFocused: $textFocused, exporting: $exporting)
        }
        .scrollBounceBehavior(.basedOnSize)
        .scrollIndicators(.never)  // a legacy scroller would narrow the content and could make the layout oscillate
        .safeAreaInset(edge: .bottom, spacing: 0) {
            ActionBar(textFocused: $textFocused)
        }
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
    var textFocused: FocusState<Bool>.Binding
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
                RecordStep()
                if contentWidth >= 900 * scale {
                    HStack(alignment: .top, spacing: gap) {
                        TextStep(textFocused: textFocused)
                        GenerateStep()
                            .frame(width: (contentWidth - 2 * gap) * 0.25)
                        ListenStep(exporting: $exporting)
                            .frame(width: (contentWidth - 2 * gap) * 0.25)
                    }
                } else {
                    TextStep(textFocused: textFocused)
                    GenerateStep()
                    ListenStep(exporting: $exporting)
                }
            }
        }
        .padding(.horizontal, padding)
        .padding(.top, 30)  // clear of the window buttons (hidden title bar)
        .padding(.bottom, 16 * scale)
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
                        NextVisitorButton()
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
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .firstTextBaseline, spacing: 24 * scale) {
                    subtitle
                    Spacer(minLength: 0)
                    DeviceStrip()
                }
                VStack(alignment: .leading, spacing: 8 * scale) {
                    subtitle
                    HStack {
                        Spacer(minLength: 0)
                        DeviceStrip()
                    }
                }
            }
            Rectangle()
                .fill(Theme.hairline)
                .frame(height: 1)
                .padding(.top, 12 * scale)
        }
    }
}

extension HeaderView {
    @ViewBuilder fileprivate var subtitle: some View {
        if !model.config.demo.subtitle.isEmpty {
            Text(model.config.demo.subtitle)
                .scaledFont(TypeSize.subtitle)
                .foregroundStyle(Theme.secondaryText)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Microphone, output and compute mode, for staff: quiet, in the header, away from the action bar.
/// They edit the same settings as Settings.
struct DeviceStrip: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        @Bindable var model = model
        HStack(alignment: .firstTextBaseline, spacing: 18 * scale) {
            MicrophoneMenu()
            OutputMenu()
            HStack(alignment: .firstTextBaseline, spacing: 6 * scale) {
                Text("Mode:")
                    .foregroundStyle(Theme.secondaryText)
                Picker("Mode", selection: $model.computeMode) {
                    Text("Exact").tag(ComputeMode.exact)
                    Text("Fast").tag(ComputeMode.fast)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.small)
                .fixedSize()
            }
        }
        .scaledFont(TypeSize.caption)
    }
}

/// The output device: the system default (followed when it changes) or a chosen one.
struct OutputMenu: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Menu {
            Button("System default" + (model.audioDevices.defaultOutputName.map { " (\($0))" } ?? "")) {
                model.chooseOutput(nil)
            }
            if !model.audioDevices.outputs.isEmpty {
                Divider()
                ForEach(model.audioDevices.outputs) { device in
                    Button(device.name) { model.chooseOutput(device) }
                }
            }
        } label: {
            Text(model.outputLabel)
                .scaledFont(TypeSize.caption, weight: .medium)
                .lineLimit(1)
        }
        .menuStyle(.button)
        .buttonStyle(.borderless)
        .fixedSize()
        .foregroundStyle(Theme.secondaryText)
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
/// The sentence to read and the recording's state; the buttons are in the action bar.
struct RecordStep: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        StepCard(number: 1, title: "Read and record", status: model.status(ofStep: 1), minHeight: 0) {
            switch model.selection {
            case .recording:
                Teleprompter(text: model.transcript, size: TypeSize.prompter * scale, active: false)
                HStack(alignment: .firstTextBaseline, spacing: 16 * scale) {
                    if let take = model.recorder.take {
                        Text("Recorded \(formatSeconds(take.duration))")
                            .foregroundStyle(Theme.secondaryText)
                        PrepareStatus()
                    } else {
                        Text("Your recording is used only for this demo and is deleted on \u{201C}Next visitor\u{201D} or when the app quits.")
                            .foregroundStyle(Theme.tertiaryText)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: 12 * scale)
                    SampleVoiceMenu(title: "No microphone? Use a sample voice")
                }
                .scaledFont(TypeSize.caption)
                RecordingNotices()
            case .preset:
                Text("Sample voice: \(model.selectedPreset?.name ?? "")")
                    .scaledFont(TypeSize.body, weight: .semibold)
                Text("\u{201C}\(model.transcript)\u{201D}")
                    .scaledFont(TypeSize.body)
                    .foregroundStyle(Theme.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 16 * scale) {
                    PrepareStatus()
                    Button("Use my own voice", action: model.useOwnVoice)
                        .buttonStyle(.borderless)
                        .foregroundStyle(Theme.accent)
                        .fontWeight(.medium)
                    SampleVoiceMenu(title: "Choose another sample voice")
                    Spacer()
                }
                .scaledFont(TypeSize.caption)
                RecordingNotices()
            }
        }
    }
}

/// The server preparing the voice (/prepare): "Preparing your voice..." with a spinner, then "Ready".
struct PrepareStatus: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        switch model.prepareState {
        case .idle:
            EmptyView()
        case .preparing:
            HStack(spacing: 6 * scale) {
                ProgressView().controlSize(.mini)
                Text("Preparing your voice...")
                    .foregroundStyle(Theme.secondaryText)
            }
        case .ready:
            Label("Ready", systemImage: "checkmark.circle.fill")
                .foregroundStyle(Theme.accent)
                .fontWeight(.medium)
        case .failed(let reason):
            HStack(alignment: .firstTextBaseline, spacing: 10 * scale) {
                Label("Could not prepare your voice: \(reason)", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(Theme.error)
                    .fixedSize(horizontal: false, vertical: true)
                if model.selection == .recording {
                    Button("Record again", action: model.startRecording)
                        .buttonStyle(.borderless)
                        .foregroundStyle(Theme.accent)
                        .fontWeight(.semibold)
                        .disabled(!model.canRecord)
                }
            }
        }
    }
}

/// Notices about the recording (too short, no speech, ...), next to it in step 1.
struct RecordingNotices: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        let notices = model.recordingNotices
        if !notices.isEmpty {
            VStack(alignment: .leading, spacing: 6 * scale) {
                ForEach(notices) { notice in
                    Label {
                        Text(notice.message)
                            .foregroundStyle(Theme.primaryText)
                            .fixedSize(horizontal: false, vertical: true)
                    } icon: {
                        Image(systemName: notice.level.systemImage)
                            .foregroundStyle(notice.level.color)
                    }
                }
            }
            .scaledFont(TypeSize.banner)
            .padding(.horizontal, 12 * scale)
            .padding(.vertical, 8 * scale)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(notices.contains { $0.level == .error } ? Theme.errorTint : Theme.warningTint,
                        in: .rect(cornerRadius: 10))
        }
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

/// Countdown and recording: the sentence fills the width, with the level and the time. Stop is
/// the primary button of the action bar, where Record was.
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

            Text("Press Stop below when you have finished the sentence. Recording stops by itself after \(Int(recorder.maxSeconds)) s.")
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
/// The text to say, prefilled; Return generates.
struct TextStep: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale
    var textFocused: FocusState<Bool>.Binding

    var body: some View {
        @Bindable var model = model
        let limit = Int(model.config.warnings.maxTextChars)
        StepCard(number: 2, title: "Type what to say", status: model.status(ofStep: 2), minHeight: 168 * scale) {
            TextField("What should your voice say?", text: $model.text, axis: .vertical)
                .textFieldStyle(.plain)
                .scaledFont(TypeSize.body)
                .lineLimit(2...3)
                .focused(textFocused)
                .onSubmit { model.performPrimaryActionFromText() }
                .padding(10 * scale)
                .frame(minHeight: 62 * scale, alignment: .topLeading)
                .background(Theme.well, in: .rect(cornerRadius: 10))
                .overlay {
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(textFocused.wrappedValue ? Theme.accent : Color.clear, lineWidth: 1.5)
                }
            .background {
                // Command-Return in the field also generates (Return is handled by onSubmit).
                Button("Generate my voice", action: model.performPrimaryActionFromText)
                    .keyboardShortcut(.return, modifiers: .command)
                    .opacity(0)
                    .frame(width: 0, height: 0)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
            HStack {
                Text("Optional. Press Return to generate.")
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
/// The state of generation; the button is in the action bar.
struct GenerateStep: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        StepCard(number: 3, title: "Generate", status: model.status(ofStep: 3), minHeight: 168 * scale) {
            if model.isSynthesizing {
                HStack(spacing: 8 * scale) {
                    ProgressView().controlSize(.small)
                    Text(model.generationStageLine)
                }
                .scaledFont(TypeSize.caption)
                .foregroundStyle(Theme.secondaryText)
            } else if model.outputIsCurrent {
                Label("Your voice is ready.", systemImage: "checkmark.circle.fill")
                    .scaledFont(TypeSize.caption, weight: .medium)
                    .foregroundStyle(Theme.accent)
            } else {
                Text(model.blockingReason ?? "Press \u{201C}Generate my voice\u{201D} below.")
                    .scaledFont(TypeSize.caption)
                    .foregroundStyle(Theme.tertiaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

// -----------------------------
// 4 Listen
// -----------------------------
/// Playback progress and the output device; Play is in the action bar.
struct ListenStep: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale
    @Binding var exporting: Bool

    var body: some View {
        StepCard(number: 4, title: "Listen", status: model.status(ofStep: 4), minHeight: 168 * scale) {
            if model.output != nil {
                PlaybackBar(player: model.player)
                if model.config.app.showSaveButton {
                    Button("Save...") { exporting = true }
                        .buttonStyle(.borderless)
                        .foregroundStyle(Theme.accent)
                }
            } else {
                Text("Your generated voice will play here.")
                    .foregroundStyle(Theme.tertiaryText)
            }
            Text(model.outputLabel)
                .foregroundStyle(Theme.tertiaryText)
                .fixedSize(horizontal: false, vertical: true)
        }
        .scaledFont(TypeSize.caption)
    }
}

// -----------------------------
// Action bar
// -----------------------------
/// The fixed action bar at the bottom of the window: the primary button in the center, always in
/// the same place, with the secondary actions of the moment on either side.
struct ActionBar: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale
    var textFocused: FocusState<Bool>.Binding

    var body: some View {
        // No GlassEffectContainer: it would blend the neutral buttons into the tinted primary one.
        HStack(spacing: 28 * scale) {
            HStack(spacing: 16 * scale) {
                if showNextVisitor {
                    // At the far edge, away from the primary button, so it is not hit by accident.
                    NextVisitorButton(compact: false)
                }
                Spacer(minLength: 0)
                leading
            }
            .frame(maxWidth: .infinity)
            PrimaryButton()
            HStack(spacing: 16 * scale) { trailing }
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 32 * scale)
        .padding(.vertical, 14 * scale)
        .frame(maxWidth: .infinity)
        .background(Theme.page.opacity(0.92))
        .overlay(alignment: .top) {
            Rectangle().fill(Theme.hairline).frame(height: 1)
        }
    }

    /// Left of the primary button: start the recording over.
    @ViewBuilder private var leading: some View {
        if canChangeRecording, model.selection == .recording, model.recorder.take != nil {
            SecondaryButton(title: "Re-record", systemImage: "arrow.counterclockwise.circle",
                            help: "Record the sentence again (\u{2318}R)", action: model.startRecording)
                .disabled(!model.canRecord)
        }
    }

    /// Right of the primary button: listen to the recording or play the voice again, edit the text.
    @ViewBuilder private var trailing: some View {
        if canChangeRecording, model.promptURL != nil {
            if !model.outputIsCurrent {
                switch model.prepareState {
                case .preparing:
                    SecondaryButton(title: "Preparing...", systemImage: "play.fill", help: "Your voice is being prepared",
                                    busy: true) {}
                        .disabled(true)
                case .ready where model.canPlayPrompt:
                    let playing = model.promptPlayer.isPlaying
                    SecondaryButton(title: playing ? "Stop" : (model.selection == .recording ? "Play my recording" : "Play sample"),
                                    systemImage: playing ? "stop.fill" : "play.fill",
                                    help: "Listen to your voice as the model hears it (cleaned and trimmed)",
                                    action: model.togglePromptPlayback)
                default:
                    EmptyView()  // nothing processed to play (a failure is shown in step 1)
                }
            }
            if !textFocused.wrappedValue {
                SecondaryButton(title: "Edit text", systemImage: "pencil", help: "Change what your voice says") {
                    textFocused.wrappedValue = true
                }
            }
        }
    }

    private var canChangeRecording: Bool {
        !model.recorder.isCapturing && !model.isSynthesizing && !model.isResetting
    }

    /// Once the generated voice has been heard, Next visitor also appears here (it is always in the header).
    private var showNextVisitor: Bool {
        canChangeRecording && model.outputIsCurrent && model.player.timesFinished > 0
    }
}

/// The one big filled button; its label follows `AppModel.primaryAction`.
struct PrimaryButton: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        let action = model.primaryAction
        if action == .generating {
            GenerationProgress()
        } else {
            button(action)
        }
    }

    private func button(_ action: AppModel.PrimaryAction) -> some View {
        Button(action: model.performPrimaryAction) {
            HStack(spacing: 10 * scale) {
                if let systemImage = action.systemImage {
                    Image(systemName: systemImage)
                } else {
                    ProgressView().controlSize(.small).tint(.white)
                }
                Text(action.title)
            }
            .scaledFont(TypeSize.action + 1, weight: .semibold)
            .frame(minWidth: 236 * scale, minHeight: 36 * scale)
            .padding(.vertical, 6 * scale)
            .contentTransition(.opacity)
        }
        .buttonStyle(.glassProminent)
        .tint(Theme.accent)
        .controlSize(.large)
        .disabled(!model.primaryActionEnabled)
        .help(action.shortcutHint.isEmpty ? action.title : "\(action.title) (\(action.shortcutHint))")
        .animation(.snappy, value: action)
    }
}

/// The primary spot while generating: the stage in plain words over a thin determinate bar, or a
/// spinner while the server cannot say how far it is. No step counts or timings.
struct GenerationProgress: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale

    var body: some View {
        VStack(spacing: 8 * scale) {
            HStack(spacing: 8 * scale) {
                if model.generationFraction == nil {
                    ProgressView().controlSize(.small)
                }
                Text(model.generationStageLine)
                    .scaledFont(TypeSize.body, weight: .semibold)
                    .foregroundStyle(Theme.primaryText)
                    .contentTransition(.opacity)
                    .animation(.snappy, value: model.generationStageLine)
            }
            if let fraction = model.generationFraction {
                ProgressTrack(fraction: fraction)
            }
        }
        .frame(width: 252 * scale, height: 48 * scale)
        .padding(.vertical, 6 * scale)
    }
}

/// A neutral glass button beside the primary one, at least 44 points high.
struct SecondaryButton: View {
    @Environment(\.uiScale) private var scale
    var title: String
    var systemImage: String
    var help: String
    var busy = false  // a small spinner instead of the icon
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            Label {
                Text(title)
            } icon: {
                if busy {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: systemImage)
                }
            }
            .scaledFont(TypeSize.body, weight: .medium)
                .frame(minHeight: 32 * scale)
                .padding(.horizontal, 4 * scale)
                .padding(.vertical, 2 * scale)
        }
        .buttonStyle(.glass)
        .controlSize(.large)
        .help(help)
    }
}

/// Deletes this visitor's recording and starts over (also File > Next visitor, Command-N). In
/// the header for staff; for visitors it is the primary button once the voice has been heard.
struct NextVisitorButton: View {
    @Environment(AppModel.self) private var model
    @Environment(\.uiScale) private var scale
    var compact = true  // the header's tinted button; false: a neutral one for the action bar

    var body: some View {
        if compact {
            Button(action: next) {
                Label("Next visitor", systemImage: "arrow.counterclockwise")
                    .scaledFont(TypeSize.body, weight: .semibold)
                    .padding(.horizontal, 4 * scale)
            }
            .buttonStyle(.bordered)
            .tint(Theme.accent)
            .controlSize(.large)
            .disabled(model.isResetting || model.recorder.isCapturing)
            .help("Delete this visitor's recording and start over (\u{2318}N)")
        } else {
            SecondaryButton(title: "Next visitor", systemImage: "arrow.counterclockwise",
                            help: "Delete this visitor's recording and start over (\u{2318}N)", action: next)
                .disabled(model.isResetting)
        }
    }

    private func next() {
        Task { await model.nextVisitor() }
    }
}
