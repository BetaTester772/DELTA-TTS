import Foundation

/// A message for the user: from the server ("warnings" of a response) or from the app's checks.
struct Notice: Codable, Equatable, Identifiable, Sendable {
    enum Level: String, Codable, Comparable, Sendable {
        case info, warning, error

        private var rank: Int {
            switch self {
            case .error: 0
            case .warning: 1
            case .info: 2
            }
        }

        static func < (a: Level, b: Level) -> Bool { a.rank < b.rank }  // errors first
    }

    var level: Level
    var code: String
    var message: String

    var id: String { "\(code)|\(message)" }
}

/// GET /health
struct Health: Decodable, Equatable, Sendable {
    var status: String  // "loading", "ready" or "error"
    var error: String?
    var device: String?
    var compute: String?
    var sampleRate: Int?
    var dryRun: Bool
    var busy: Bool?
    var backend: BackendStatus?  // the current mode's; nil when the server or the pipeline does not report it
    var fastBackend: BackendStatus?  // the fast mode's, also while warming up in the background in exact mode
    var timing: [String: ModeTiming]?  // per compute mode, measured on this machine

    enum CodingKeys: String, CodingKey {
        case status, error, device, compute, busy, backend, timing
        case fastBackend = "fast_backend"
        case sampleRate = "sample_rate"
        case dryRun = "dry_run"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        status = try c.decode(String.self, forKey: .status)
        error = try c.decodeIfPresent(String.self, forKey: .error)
        device = try c.decodeIfPresent(String.self, forKey: .device)
        compute = try c.decodeIfPresent(String.self, forKey: .compute)
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate)
        dryRun = try c.decodeIfPresent(Bool.self, forKey: .dryRun) ?? false
        busy = try c.decodeIfPresent(Bool.self, forKey: .busy)
        // An optional display: a field the app does not understand must not break /health.
        backend = try? c.decodeIfPresent(BackendStatus.self, forKey: .backend)
        fastBackend = try? c.decodeIfPresent(BackendStatus.self, forKey: .fastBackend)
        timing = try? c.decodeIfPresent([String: ModeTiming].self, forKey: .timing)
    }
}

/// Clean generations of one compute mode on the server (generation stages only, no fallback,
/// backend ready).
struct ModeTiming: Decodable, Equatable, Sendable {
    var generationSeconds: Double?  // median
    var generationPerOutputSecond: Double?  // median, comparable across texts
    var count: Int

    enum CodingKeys: String, CodingKey {
        case generationSeconds = "generation_s"
        case generationPerOutputSecond = "generation_per_output_s"
        case count = "n"
    }
}

/// The generation backend's readiness (`DeltaTTSPipeline.backend_status()`), e.g. the fast
/// mode's Neural Engine functions compiling in the background. Every field is optional.
struct BackendStatus: Decodable, Equatable, Sendable {
    var backend: String?  // "neural_engine", "float16" or "float32"; nil before the fast backend is built
    var state: String?  // "not_started", "idle", "loading", "compiling", "ready" or "failed"
    var readyBuckets: Int?
    var totalBuckets: Int?
    var maxReadyLen: Int?
    var failedBuckets: Int?  // how many
    var error: String?

    enum CodingKeys: String, CodingKey {
        case backend, state, error
        case readyBuckets = "ready_buckets"
        case totalBuckets = "total_buckets"
        case maxReadyLen = "max_ready_len"
        case failedBuckets = "failed_buckets"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        backend = try? c.decodeIfPresent(String.self, forKey: .backend)
        state = try? c.decodeIfPresent(String.self, forKey: .state)
        readyBuckets = try? c.decodeIfPresent(Int.self, forKey: .readyBuckets)
        totalBuckets = try? c.decodeIfPresent(Int.self, forKey: .totalBuckets)
        maxReadyLen = try? c.decodeIfPresent(Int.self, forKey: .maxReadyLen)
        failedBuckets = (try? c.decodeIfPresent([AnyJSON].self, forKey: .failedBuckets))?.count
        error = try? c.decodeIfPresent(String.self, forKey: .error)
    }
}

/// Any JSON value, ignored (to count the elements of an array whose shape may change).
private struct AnyJSON: Decodable {
    init(from decoder: Decoder) throws {}
}

/// POST /synthesize body. Keys are encoded as snake_case. The prompt is either `promptId` (from
/// /prepare) or the audio itself.
struct SynthesisRequest: Encodable, Sendable {
    var text: String
    var promptText: String
    var promptId: String?
    var promptWavBase64: String?
    var promptWavName: String?
    var lengthScale: Double
    var numSteps: Int
    var temperature: Double
    var topP: Double
    var tShift: Double
    var compute: ComputeMode
    var enhancePrompt: Bool
    var trimPrompt: Bool
    var trimOutput: Bool
    var warnings: DemoConfig.Thresholds
}

/// GET /progress: the request in flight on the server.
struct ProgressSnapshot: Decodable, Equatable, Sendable {
    var state: String  // "idle", "preparing" or "generating"
    var stage: String?  // "prompt", "tokens", "acoustic" or "vocoder"
    var fraction: Double?  // 0...1, nil when indeterminate
    var requestId: Int

    enum CodingKeys: String, CodingKey {
        case state, stage, fraction
        case requestId = "request_id"
    }
}

/// POST /prepare body: the prompt part of a synthesis request.
struct PrepareRequest: Encodable, Sendable {
    var promptText: String
    var promptWavBase64: String
    var promptWavName: String
    var enhancePrompt: Bool
    var trimPrompt: Bool
    var trimOutput: Bool
    var warnings: DemoConfig.Thresholds
}

/// POST /prepare response.
struct PrepareResponse: Decodable, Sendable {
    var promptId: String
    var durationSeconds: Double
    var audioWavBase64: String?  // the processed prompt: what the model uses
    var warnings: [Notice]

    enum CodingKeys: String, CodingKey {
        case warnings
        case promptId = "prompt_id"
        case durationSeconds = "duration_s"
        case audioWavBase64 = "audio_wav_base64"
    }
}

/// POST /synthesize response.
struct SynthesisResponse: Decodable, Sendable {
    var audioWavBase64: String
    var sampleRate: Int
    var durationSeconds: Double
    var estimatedDurationSeconds: Double?
    var elapsedSeconds: Double
    var rtf: Double?
    var compute: String?
    var fellBack: Bool?
    var dryRun: Bool?
    var warnings: [Notice]

    enum CodingKeys: String, CodingKey {
        case rtf, compute, warnings
        case audioWavBase64 = "audio_wav_base64"
        case sampleRate = "sample_rate"
        case durationSeconds = "duration_s"
        case estimatedDurationSeconds = "estimated_duration_s"
        case elapsedSeconds = "elapsed_s"
        case fellBack = "fell_back"
        case dryRun = "dry_run"
    }
}

/// POST /reset response.
struct ResetResponse: Decodable, Sendable {
    var status: String
    var removedUploads: Int?
    var clearedPromptFeatures: Int?

    enum CodingKeys: String, CodingKey {
        case status
        case removedUploads = "removed_uploads"
        case clearedPromptFeatures = "cleared_prompt_features"
    }
}

/// Body of an error response.
private struct ErrorResponse: Decodable {
    var error: String
    var code: String?
    var warnings: [Notice]?
}

enum ServerError: LocalizedError {
    case unreachable(URL, String)
    case http(status: Int, message: String, notices: [Notice], code: String?)
    case invalidResponse(String)

    var errorDescription: String? {
        switch self {
        case .unreachable(let url, let reason): "Cannot reach the server at \(url.absoluteString) (\(reason))."
        case .http(let status, let message, _, _): "\(message) (HTTP \(status))"
        case .invalidResponse(let reason): "Unexpected response from the server: \(reason)"
        }
    }

    /// Notices the server sent along with the error.
    var notices: [Notice] {
        if case .http(_, _, let notices, _) = self { return notices }
        return []
    }

    /// The server's error code (e.g. "prompt_unknown").
    var code: String? {
        if case .http(_, _, _, let code) = self { return code }
        return nil
    }
}

/// The JSON API of demo/server.py.
struct ServerClient: Sendable {
    var baseURL: URL

    func health() async throws -> Health {
        var request = URLRequest(url: baseURL.appending(path: "health"))
        request.timeoutInterval = 3
        return try await send(request, as: Health.self)
    }

    func synthesize(_ body: SynthesisRequest) async throws -> SynthesisResponse {
        try await post("synthesize", body, timeout: 900, as: SynthesisResponse.self)  // long texts on a slow device
    }

    /// Switch the server's compute mode now (it starts warming up the fast backend).
    func setCompute(_ mode: ComputeMode) async throws -> Health {
        try await post("compute", ["compute": mode.rawValue], timeout: 120, as: Health.self)
    }

    func progress() async throws -> ProgressSnapshot {
        var request = URLRequest(url: baseURL.appending(path: "progress"))
        request.timeoutInterval = 2
        return try await send(request, as: ProgressSnapshot.self)
    }

    /// Process the prompt ahead of synthesis (VAD trim, enhancement, features).
    func prepare(_ body: PrepareRequest) async throws -> PrepareResponse {
        try await post("prepare", body, timeout: 300, as: PrepareResponse.self)  // the first call loads RE-USE
    }

    private func post<Body: Encodable, T: Decodable>(_ path: String, _ body: Body, timeout: TimeInterval,
                                                    as type: T.Type) async throws -> T {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        var request = URLRequest(url: baseURL.appending(path: path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(body)
        request.timeoutInterval = timeout
        return try await send(request, as: T.self)
    }

    /// Make the server forget the previous visitor (uploaded prompts, cached prompt features).
    func reset(timeout: TimeInterval = 10) async throws -> ResetResponse {
        var request = URLRequest(url: baseURL.appending(path: "reset"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("{}".utf8)
        request.timeoutInterval = timeout
        return try await send(request, as: ResetResponse.self)
    }

    /// `reset` that blocks for at most `timeout` seconds, for use while the app quits (the main
    /// run loop no longer serves async work then). The response is not checked.
    func resetBlocking(timeout: TimeInterval) {
        var request = URLRequest(url: baseURL.appending(path: "reset"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("{}".utf8)
        request.timeoutInterval = timeout
        let done = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: request) { _, _, _ in done.signal() }.resume()
        _ = done.wait(timeout: .now() + timeout)
    }

    private func send<T: Decodable>(_ request: URLRequest, as type: T.Type) async throws -> T {
        let result: (Data, URLResponse)
        do {
            result = try await URLSession.shared.data(for: request)
        } catch let error as URLError {
            throw ServerError.unreachable(baseURL, error.localizedDescription)
        }
        let (data, response) = result
        guard let http = response as? HTTPURLResponse else { throw ServerError.invalidResponse("not HTTP") }
        guard http.statusCode == 200 else {
            if let body = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
                throw ServerError.http(status: http.statusCode, message: body.error, notices: body.warnings ?? [], code: body.code)
            }
            let text = String(decoding: data.prefix(300), as: UTF8.self)
            throw ServerError.http(status: http.statusCode, message: text.isEmpty ? "No details" : text, notices: [], code: nil)
        }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw ServerError.invalidResponse(error.localizedDescription)
        }
    }
}
