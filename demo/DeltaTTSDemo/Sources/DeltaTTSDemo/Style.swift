import SwiftUI
import UniformTypeIdentifiers

/// A light theme: a very light neutral page, white cards with hairline borders, near-black text,
/// one blue accent, and muted semantic colors.
enum Theme {
    static let accent = Color(hex: 0x3253DC)
    static let accentTint = Color(hex: 0xEDF1FD)  // selected and current backgrounds
    static let page = Color(hex: 0xF7F8FA)
    static let card = Color.white
    static let well = Color(hex: 0xF2F4F7)  // text areas and the sentence box inside cards
    static let hairline = Color(hex: 0xE3E6EB)
    static let primaryText = Color(hex: 0x111418)
    static let secondaryText = Color(hex: 0x4F5661)
    static let tertiaryText = Color(hex: 0x858C97)
    static let idle = Color(hex: 0xB4BAC3)
    static let warning = Color(hex: 0xA86A12)  // muted amber, readable on white
    static let warningTint = Color(hex: 0xFDF6EA)
    static let error = Color(hex: 0xB93A35)  // muted red
    static let errorTint = Color(hex: 0xFCEFEE)

    static let cornerRadius: CGFloat = 16
}

extension Color {
    init(hex: UInt32) {
        self.init(red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255)
    }
}

extension EnvironmentValues {
    /// Scale of the main screen's type (`app.ui_scale`).
    @Entry var uiScale: Double = 1
}

/// Type sizes of the main screen at `ui_scale` 1, in points, tuned for a 14-inch MacBook Pro
/// (1512 x 982 points). Every size and spacing of the main screen is multiplied by `ui_scale`.
enum TypeSize {
    static let banner: CGFloat = 14
    static let caption: CGFloat = 14
    static let venue: CGFloat = 13
    static let body: CGFloat = 16
    static let subtitle: CGFloat = 16
    static let instruction: CGFloat = 18
    static let action: CGFloat = 18
    static let stepTitle: CGFloat = 19
    static let title: CGFloat = 32
    static let prompter: CGFloat = 40  // in step 1
    static let prompterFocus: CGFloat = 46  // while recording; up to 30% larger on wide screens
}

/// Spacing of the main screen at `ui_scale` 1, in points.
enum Spacing {
    static let card: CGFloat = 18  // inside a card
    static let gap: CGFloat = 16  // between cards and sections
    static let item: CGFloat = 12  // between items in a card
}

private struct ScaledFont: ViewModifier {
    @Environment(\.uiScale) private var scale
    var size: CGFloat
    var weight: Font.Weight

    func body(content: Content) -> some View {
        content.font(.system(size: size * scale, weight: weight))
    }
}

extension View {
    /// The system font (SF Pro) at `size` points times the main screen's `ui_scale`.
    func scaledFont(_ size: CGFloat, weight: Font.Weight = .regular) -> some View {
        modifier(ScaledFont(size: size, weight: weight))
    }

    /// The button style of a main-screen action: filled with the accent only for the one action
    /// to take now, else neutral glass. (Glass buttons take any inherited tint as their fill,
    /// so the main screen sets no tint of its own.)
    @ViewBuilder func actionStyle(prominent: Bool) -> some View {
        if prominent {
            buttonStyle(.glassProminent).tint(Theme.accent)
        } else {
            buttonStyle(.glass)
        }
    }

    /// A flat white card with a hairline border and a faint shadow.
    func card(highlighted: Bool = false) -> some View {
        background(Theme.card, in: .rect(cornerRadius: Theme.cornerRadius))
            .overlay {
                RoundedRectangle(cornerRadius: Theme.cornerRadius)
                    .strokeBorder(highlighted ? Theme.accent : Theme.hairline, lineWidth: highlighted ? 1.5 : 1)
            }
            .shadow(color: .black.opacity(0.04), radius: 10, y: 2)
    }
}

/// Label of a main-screen button (about 44 points high at `ui_scale` 1 with `.controlSize(.large)`).
struct ActionLabel: View {
    @Environment(\.uiScale) private var scale
    var title: String
    var systemImage: String?

    var body: some View {
        Group {
            if let systemImage {
                Label(title, systemImage: systemImage)
            } else {
                Text(title)
            }
        }
        .scaledFont(TypeSize.action, weight: .semibold)
        .padding(.horizontal, 6 * scale)
        .padding(.vertical, 4 * scale)
    }
}

enum StepStatus {
    case upcoming, current, done
}

/// One numbered step of the main screen; the current step is outlined in the accent.
struct StepCard<Content: View>: View {
    @Environment(\.uiScale) private var scale
    var number: Int
    var title: String
    var status: StepStatus
    var minHeight: CGFloat = 0
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.item * scale) {
            HStack(spacing: 10 * scale) {
                StepBadge(number: number, status: status)
                Text(title)
                    .scaledFont(TypeSize.stepTitle, weight: .semibold)
                    .foregroundStyle(status == .upcoming ? Theme.tertiaryText : Theme.primaryText)
            }
            content
        }
        .padding(Spacing.card * scale)
        .frame(maxWidth: .infinity, minHeight: minHeight, alignment: .topLeading)
        .card(highlighted: status == .current)
    }
}

struct StepBadge: View {
    @Environment(\.uiScale) private var scale
    var number: Int
    var status: StepStatus

    var body: some View {
        ZStack {
            switch status {
            case .current:
                Circle().fill(Theme.accent)
                Text("\(number)").foregroundStyle(.white)
            case .done:
                Circle().fill(Theme.accentTint)
                Image(systemName: "checkmark").foregroundStyle(Theme.accent)
            case .upcoming:
                Circle().strokeBorder(Theme.idle, lineWidth: 1.5)
                Text("\(number)").foregroundStyle(Theme.tertiaryText)
            }
        }
        .font(.system(size: 14 * scale, weight: .semibold))
        .frame(width: 28 * scale, height: 28 * scale)
    }
}

/// The sentence to read aloud: near-black on a light well, as large as the place allows.
struct Teleprompter: View {
    var text: String
    var size: CGFloat  // points, already scaled
    var active: Bool  // counting down or recording

    var body: some View {
        Text(text.isEmpty ? "No sentence to read. Set one in Settings > Demo." : text)
            .font(.system(size: size, weight: .semibold))
            .foregroundStyle(Theme.primaryText)
            .lineSpacing(size * 0.12)
            .multilineTextAlignment(active ? .center : .leading)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: active ? .center : .leading)
            .padding(.horizontal, size * 0.5)
            .padding(.vertical, size * 0.3)
            .background(active ? Theme.card : Theme.well, in: .rect(cornerRadius: 14))
            .overlay {
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(active ? Theme.accent : Color.clear, lineWidth: 2)
            }
    }
}

/// Input level as a row of segments.
struct LevelMeter: View {
    var level: Double  // 0...1
    var segments = 32

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<segments, id: \.self) { index in
                RoundedRectangle(cornerRadius: 3)
                    .fill(Double(index) < level * Double(segments) ? Theme.accent : Theme.hairline)
            }
        }
        .animation(.linear(duration: 0.05), value: level)
    }
}

/// Playback position: a thin rounded track with a blue fill and elapsed / total time. It is
/// redrawn every display frame from the player's own position, without implicit animation, so
/// it moves exactly with the audio: from 0 when playback starts to the end when it ends, where it
/// stays until the next playback. Clicking it plays from that point.
struct PlaybackBar: View {
    @Environment(\.uiScale) private var scale
    var player: AudioPlayer

    var body: some View {
        TimelineView(.animation(paused: !player.isPlaying)) { _ in
            let progress = player.progress
            let total = player.duration.rounded()
            let elapsed = player.atEnd ? total : min(player.currentTime.rounded(.down), total)
            HStack(spacing: 10 * scale) {
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Theme.hairline)
                        Capsule().fill(Theme.accent)
                            .frame(width: geometry.size.width * progress)
                    }
                    .frame(height: 6 * scale)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .contentShape(.rect)
                    .onTapGesture { location in
                        guard geometry.size.width > 0 else { return }
                        player.play(from: location.x / geometry.size.width * player.duration)
                    }
                }
                .frame(height: 22 * scale)
                .help("Click to play from there")
                Text("\(Self.clock(elapsed)) / \(Self.clock(total))")
                    .monospacedDigit()
                    .foregroundStyle(Theme.tertiaryText)
                    .fixedSize()
            }
            .transaction { $0.animation = nil }
        }
    }

    /// Seconds as "0:04".
    static func clock(_ seconds: Double) -> String {
        let whole = max(0, Int(seconds))
        return String(format: "%d:%02d", whole / 60, whole % 60)
    }
}

extension Notice.Level {
    var color: Color {
        switch self {
        case .info: Theme.accent
        case .warning: Theme.warning
        case .error: Theme.error
        }
    }

    var tint: Color {
        switch self {
        case .info: Theme.accentTint
        case .warning: Theme.warningTint
        case .error: Theme.errorTint
        }
    }

    var systemImage: String {
        switch self {
        case .info: "info.circle"
        case .warning: "exclamationmark.triangle"
        case .error: "xmark.octagon"
        }
    }
}

/// The synthesized WAV for `fileExporter`.
struct WAVDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.wav] }

    var data: Data

    init(data: Data) {
        self.data = data
    }

    init(configuration: ReadConfiguration) throws {
        data = configuration.file.regularFileContents ?? Data()
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}

/// Seconds as "3.2 s".
func formatSeconds(_ seconds: Double) -> String {
    String(format: "%.1f s", seconds)
}
