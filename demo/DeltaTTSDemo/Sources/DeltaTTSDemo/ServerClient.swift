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

    enum CodingKeys: String, CodingKey {
        case status, error, device, compute, busy
        case sampleRate = "sample_rate"
        case dryRun = "dry_run"
    }
}

/// POST /synthesize body. Keys are encoded as snake_case.
struct SynthesisRequest: Encodable, Sendable {
    var text: String
    var promptText: String
    var promptWavBase64: String
    var promptWavName: String
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

/// Body of an error response.
private struct ErrorResponse: Decodable {
    var error: String
    var code: String?
    var warnings: [Notice]?
}

enum ServerError: LocalizedError {
    case unreachable(URL, String)
    case http(status: Int, message: String, notices: [Notice])
    case invalidResponse(String)

    var errorDescription: String? {
        switch self {
        case .unreachable(let url, let reason): "Cannot reach the server at \(url.absoluteString) (\(reason))."
        case .http(let status, let message, _): "\(message) (HTTP \(status))"
        case .invalidResponse(let reason): "Unexpected response from the server: \(reason)"
        }
    }

    /// Notices the server sent along with the error.
    var notices: [Notice] {
        if case .http(_, _, let notices) = self { return notices }
        return []
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
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        var request = URLRequest(url: baseURL.appending(path: "synthesize"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(body)
        request.timeoutInterval = 900  // long texts on a slow device
        return try await send(request, as: SynthesisResponse.self)
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
                throw ServerError.http(status: http.statusCode, message: body.error, notices: body.warnings ?? [])
            }
            let text = String(decoding: data.prefix(300), as: UTF8.self)
            throw ServerError.http(status: http.statusCode, message: text.isEmpty ? "No details" : text, notices: [])
        }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw ServerError.invalidResponse(error.localizedDescription)
        }
    }
}
