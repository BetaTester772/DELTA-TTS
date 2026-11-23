import Foundation

/// Compute mode of DELTA-TTS on the server.
enum ComputeMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case exact, fast

    var id: String { rawValue }
    var title: String { self == .exact ? "Exact (float32)" : "Fast (float16)" }
}

/// The demo config: the sections of demo/config.json, shared with demo/server.py, plus the
/// app's own settings (`app`). Keys are snake_case in JSON.
struct DemoConfig: Codable, Equatable, Sendable {
    struct Server: Codable, Equatable, Sendable {
        var host = "127.0.0.1"
        var port = 8765
        var backboneDir = "pretrained_models/Fun-CosyVoice3-0.5B-2512"
        var checkpoint = "pretrained_models/DELTA-TTS/delta_tts.pt"
        var reuseDir = "nvidia/NVIDIA-NemotronLabs-RE-USE"
        var device: String?
        var compute = ComputeMode.exact
        var allowedOrigins: [String] = []
    }

    struct Preset: Codable, Equatable, Identifiable, Sendable {
        var id = UUID()
        var name: String
        var wav: String  // relative to the repository root, or absolute
        var transcript: String

        enum CodingKeys: String, CodingKey { case name, wav, transcript }
    }

    struct Synthesis: Codable, Equatable, Sendable {
        var lengthScale = 1.0
        var numSteps = 16
        var temperature = 0.0
        var topP = 0.8
        var tShift = 0.3
        var compute = ComputeMode.exact
    }

    struct Processing: Codable, Equatable, Sendable {
        var enhancePrompt = true  // RE-USE on the visitor's recording
        var trimPrompt = true  // Silero VAD on the visitor's recording
        var trimOutput = false
    }

    struct Thresholds: Codable, Equatable, Sendable {
        var minPromptSeconds = 3.0
        var maxPromptSeconds = 30.0
        var minPromptSampleRate = 16000.0
        var maxTextChars = 400.0
        var maxOutputSeconds = 30.0
    }

    /// What a visitor starts with (the app only; the server ignores this section).
    struct Demo: Codable, Equatable, Sendable {
        var title = "DELTA-TTS"
        var subtitle = "Adapting Autoregressive Model into Diffusion Language Model for Text-to-Speech"
        var venue = "NeurIPS 2026"
        var defaultTranscript = "To be, or not to be, that is the question"  // read aloud while recording
        var defaultText = "Hello, this is your voice."
        /// Measured speedup of the fast mode, for the Mode tooltip when this Mac has no live
        /// measurement yet (null: none).
        var fastSpeedupPercent: Double?
    }

    /// Microphone recording of the prompt (the app only).
    struct Recording: Codable, Equatable, Sendable {
        var maxSeconds = 12.0  // recording stops by itself after this
        var countdownSeconds = 3
        var sampleRate = 24000.0  // mono 16-bit WAV
        var trimSilence = true  // cut leading and trailing silence before upload
        var inputDeviceUid = ""  // empty: the system default input
        var inputDeviceName = ""  // name of that device, for messages when it is missing
    }

    /// Settings of the app only; the server ignores this section.
    struct App: Codable, Equatable, Sendable {
        var serverUrl = ""  // empty: http://<server.host>:<server.port>
        var repoPath = ""  // DELTA-TTS checkout, for presets, defaults and starting the server
        var uvPath = ""  // empty: search the usual install locations
        var launchDryRun = false
        var uiScale = 1.0  // text and control size on the main screen
        var showSaveButton = false  // saved files are not deleted by Next visitor
        var outputDeviceUid = ""  // empty: the system default output
        var outputDeviceName = ""  // name of that device, for messages
    }

    var server = Server()
    var presets: [Preset] = []
    var synthesis = Synthesis()
    var processing = Processing()
    var warnings = Thresholds()
    var demo = Demo()
    var recording = Recording()
    var app = App()
}

extension DemoConfig {
    /// The server URL to use: `app.serverUrl`, or one built from `server.host` and `server.port`.
    var serverURL: URL? {
        let custom = app.serverUrl.trimmingCharacters(in: .whitespaces)
        if !custom.isEmpty { return URL(string: custom) }
        return URL(string: "http://\(Self.clientHost(server.host)):\(server.port)")
    }

    var defaultServerURLString: String { "http://\(Self.clientHost(server.host)):\(server.port)" }

    private static func clientHost(_ host: String) -> String {
        switch host {
        case "", "0.0.0.0", "::": "127.0.0.1"
        case let h where h.contains(":"): "[\(h)]"
        default: host
        }
    }

    /// A preset or prompt path, resolved against the repository unless absolute.
    func resolve(_ path: String) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") { return URL(fileURLWithPath: expanded) }
        return URL(fileURLWithPath: app.repoPath, isDirectory: true).appending(path: expanded)
    }

    /// `path` relative to the repository when it lies inside it, else absolute.
    func repoRelative(_ url: URL) -> String {
        let root = URL(fileURLWithPath: app.repoPath, isDirectory: true).standardizedFileURL.path
        let path = url.standardizedFileURL.path
        if !app.repoPath.isEmpty, path.hasPrefix(root + "/") { return String(path.dropFirst(root.count + 1)) }
        return path
    }

    // -----------------------------
    // JSON
    // -----------------------------
    static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    /// Decode a config file. Keys it lacks keep the built-in defaults, so that older or partial
    /// files still load.
    static func decode(_ data: Data) throws -> DemoConfig {
        let defaults = try JSONSerialization.jsonObject(with: makeEncoder().encode(DemoConfig()))
        guard let file = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let base = defaults as? [String: Any] else {
            throw ConfigError.notAnObject
        }
        let merged = try JSONSerialization.data(withJSONObject: merge(base, file))
        return try makeDecoder().decode(DemoConfig.self, from: merged)
    }

    func encoded() throws -> Data { try Self.makeEncoder().encode(self) }

    /// `overlay` on top of `base`; nested objects are merged, everything else replaced.
    private static func merge(_ base: [String: Any], _ overlay: [String: Any]) -> [String: Any] {
        var result = base
        for (key, value) in overlay {
            if let nested = value as? [String: Any], let baseNested = base[key] as? [String: Any] {
                result[key] = merge(baseNested, nested)
            } else {
                result[key] = value
            }
        }
        return result
    }
}

enum ConfigError: LocalizedError {
    case notAnObject
    case noRepository(String)

    var errorDescription: String? {
        switch self {
        case .notAnObject: "The config file is not a JSON object."
        case .noRepository(let path):
            path.isEmpty
                ? "Set the DELTA-TTS repository path in Settings."
                : "No demo/config.json in \(path). Set the DELTA-TTS repository path in Settings."
        }
    }
}

/// Where the config lives: the user's copy in Application Support and the repository defaults.
enum ConfigStore {
    /// ~/Library/Application Support/DeltaTTSDemo/config.json, or $DELTA_TTS_DEMO_CONFIG (e.g. one
    /// profile per screen, or a test setup).
    static var userConfigURL: URL {
        if let path = ProcessInfo.processInfo.environment["DELTA_TTS_DEMO_CONFIG"], !path.isEmpty {
            return URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        }
        return URL.applicationSupportDirectory.appending(path: "DeltaTTSDemo/config.json")
    }

    /// The user's config, or nil when there is none yet.
    static func loadUser() throws -> DemoConfig? {
        guard FileManager.default.fileExists(atPath: userConfigURL.path) else { return nil }
        return try DemoConfig.decode(Data(contentsOf: userConfigURL))
    }

    static func save(_ config: DemoConfig) throws {
        try FileManager.default.createDirectory(at: userConfigURL.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try config.encoded().write(to: userConfigURL, options: .atomic)
    }

    /// demo/config.json of the repository at `repoPath`, with `app.repoPath` set.
    static func loadRepoDefaults(repoPath: String) throws -> DemoConfig {
        guard isRepository(repoPath) else { throw ConfigError.noRepository(repoPath) }
        var config = try DemoConfig.decode(Data(contentsOf: repoConfigURL(repoPath)))
        config.app.repoPath = repoPath
        return config
    }

    static func repoConfigURL(_ repoPath: String) -> URL {
        URL(fileURLWithPath: repoPath, isDirectory: true).appending(path: "demo/config.json")
    }

    static func isRepository(_ path: String) -> Bool {
        !path.isEmpty && FileManager.default.fileExists(atPath: repoConfigURL(path).path)
    }

    /// The checkout this app was built from (the package is demo/DeltaTTSDemo), if it still exists.
    static func builtFromRepository() -> String? {
        var url = URL(fileURLWithPath: #filePath)  // <repo>/demo/DeltaTTSDemo/Sources/DeltaTTSDemo/Config.swift
        for _ in 0..<5 { url.deleteLastPathComponent() }
        return isRepository(url.path) ? url.path : nil
    }
}
