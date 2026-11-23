import Foundation
import Observation

/// demo/server.py started by the app as `uv run --extra mac --extra audio python demo/server.py`,
/// with its output kept for the log view. The extras matter: without `mac`, macOS has no
/// onnxruntime (the server cannot start) and no coremltools (no Neural Engine); without `audio`,
/// no Silero VAD or RE-USE. `uv run` installs them if they are missing.
@MainActor
@Observable
final class ServerProcess {
    private(set) var isRunning = false
    private(set) var log = ""
    private var process: Process?

    private static let maxLogCharacters = 200_000

    /// Start the server in `repoPath` with the given config file.
    func start(repoPath: String, uvPath: String, configURL: URL, dryRun: Bool) throws {
        guard process == nil else { return }
        guard ConfigStore.isRepository(repoPath) else { throw ConfigError.noRepository(repoPath) }
        let uv = try Self.findUV(configured: uvPath)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: uv)
        process.arguments = ["run", "--extra", "mac", "--extra", "audio", "python", "demo/server.py",
                             "--config", configURL.path] + (dryRun ? ["--dry-run"] : [])
        process.currentDirectoryURL = URL(fileURLWithPath: repoPath, isDirectory: true)
        // An app started from Finder gets a minimal PATH; uv and FFmpeg are usually in Homebrew's.
        var environment = ProcessInfo.processInfo.environment
        let path = [(uv as NSString).deletingLastPathComponent, "/opt/homebrew/bin", "/usr/local/bin",
                    environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"]
        environment["PATH"] = path.joined(separator: ":")
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {  // end of output
                handle.readabilityHandler = nil
                return
            }
            guard let self else { return }
            let text = String(decoding: data, as: UTF8.self)
            Task { @MainActor in self.append(text) }
        }
        process.terminationHandler = { [weak self] process in
            let status = process.terminationStatus
            guard let self else { return }
            Task { @MainActor in self.didExit(status: status) }
        }

        append("$ uv \(process.arguments?.joined(separator: " ") ?? "")\n")
        try process.run()
        self.process = process
        isRunning = true
    }

    /// Ask the server to shut down (SIGINT, then SIGTERM after a few seconds).
    func stop() {
        guard let process, process.isRunning else { return }
        let id = ObjectIdentifier(process)
        send(signal: SIGINT)
        Task {
            try? await Task.sleep(for: .seconds(5))
            if let current = self.process, ObjectIdentifier(current) == id { self.send(signal: SIGTERM) }
        }
    }

    /// Stop the server right away, e.g. when the app quits.
    func terminate() {
        send(signal: SIGTERM)
    }

    func clearLog() { log = "" }

    /// Send `signal` to the process group of uv (uv and the server) when uv leads one, else to uv.
    private func send(signal: Int32) {
        guard let process, process.isRunning else { return }
        let pid = process.processIdentifier
        if kill(-pid, signal) != 0 { kill(pid, signal) }
    }

    private func append(_ text: String) {
        log += text
        if log.count > Self.maxLogCharacters { log = String(log.suffix(Self.maxLogCharacters / 2)) }
    }

    private func didExit(status: Int32) {
        append("[server exited with status \(status)]\n")
        process = nil
        isRunning = false
    }

    /// The configured uv, or the first one found in the usual install locations and PATH.
    static func findUV(configured: String) throws -> String {
        let fileManager = FileManager.default
        let custom = (configured.trimmingCharacters(in: .whitespaces) as NSString).expandingTildeInPath
        if !custom.isEmpty {
            guard fileManager.isExecutableFile(atPath: custom) else { throw ServerProcessError.uvNotFound(custom) }
            return custom
        }
        let home = fileManager.homeDirectoryForCurrentUser.path
        let candidates = ["/opt/homebrew/bin/uv", "/usr/local/bin/uv", "\(home)/.local/bin/uv", "\(home)/.cargo/bin/uv"]
            + (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":").map { "\($0)/uv" }
        guard let uv = candidates.first(where: { fileManager.isExecutableFile(atPath: $0) }) else {
            throw ServerProcessError.uvNotFound(nil)
        }
        return uv
    }
}

enum ServerProcessError: LocalizedError {
    case uvNotFound(String?)

    var errorDescription: String? {
        switch self {
        case .uvNotFound(let path?): "uv is not an executable at \(path). Fix the uv path in Settings."
        case .uvNotFound(nil): "uv was not found. Install it (brew install uv) or set its path in Settings."
        }
    }
}
