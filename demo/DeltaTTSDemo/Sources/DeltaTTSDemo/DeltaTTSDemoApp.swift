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
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1280, height: 800)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Next visitor") {
                    Task { await appDelegate.model.nextVisitor() }
                }
                .keyboardShortcut("n", modifiers: .command)
            }
        }

        Settings {
            SettingsView()
                .environment(appDelegate.model)
        }
    }
}

/// Owns the model, brings the app to the front when it is started from a terminal, and on quit
/// deletes the recordings, asks the server to forget the visitor and stops a server it started.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private(set) lazy var model = AppModel()  // created on first use, on the main actor

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        model.cleanUpForQuit()
    }
}
