import SwiftUI
import UniformTypeIdentifiers

/// A colorful backdrop, so that the Liquid Glass surfaces above it have something to refract.
struct Backdrop: View {
    var body: some View {
        ZStack {
            LinearGradient(colors: [.indigo.opacity(0.45), .teal.opacity(0.30), .purple.opacity(0.40)],
                           startPoint: .topLeading, endPoint: .bottomTrailing)
            Circle().fill(.cyan.opacity(0.35)).frame(width: 420, height: 420).blur(radius: 120).offset(x: -220, y: -260)
            Circle().fill(.pink.opacity(0.30)).frame(width: 380, height: 380).blur(radius: 120).offset(x: 260, y: 280)
        }
        .ignoresSafeArea()
    }
}

/// A titled section on a Liquid Glass panel.
struct GlassCard<Content: View>: View {
    var title: String
    var systemImage: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: systemImage)
                .font(.headline)
            content
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(.regular, in: .rect(cornerRadius: 22))
    }
}

extension Notice.Level {
    var color: Color {
        switch self {
        case .info: .blue
        case .warning: .orange
        case .error: .red
        }
    }

    var systemImage: String {
        switch self {
        case .info: "info.circle.fill"
        case .warning: "exclamationmark.triangle.fill"
        case .error: "xmark.octagon.fill"
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

/// Seconds as "3.20 s".
func formatSeconds(_ seconds: Double) -> String {
    String(format: "%.2f s", seconds)
}
