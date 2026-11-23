import AppKit
import SwiftUI

@main
struct DeltaTTSDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("DELTA-TTS", id: "main") {
            ContentView()
                .environment(appDelegate.model)
        }
        .defaultSize(width: 760, height: 860)
    }
}

/// Owns the model, brings the app to the front when it is started from a terminal (it is not
/// an app bundle), and stops a server the app started when it quits.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private(set) lazy var model = AppModel()  // created on first use, on the main actor

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        model.serverProcess.terminate()
    }
}
