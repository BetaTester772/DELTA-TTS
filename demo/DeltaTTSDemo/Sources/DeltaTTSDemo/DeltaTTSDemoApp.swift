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
            CommandGroup(replacing: .printItem) {}  // Command-P plays
            CommandMenu("Demo") {
                // Space is not a key equivalent here: it would fire while typing. AppDelegate
                // handles it when no text field has focus.
                Button("Next step (Space)") { appDelegate.model.performPrimaryAction() }
                Divider()
                Button("Record or stop") { appDelegate.model.toggleRecording() }
                    .keyboardShortcut("r", modifiers: .command)
                Button("Generate my voice") { Task { await appDelegate.model.synthesize() } }
                    .keyboardShortcut("g", modifiers: .command)
                Button("Play or stop") { appDelegate.model.toggleOutputPlayback() }
                    .keyboardShortcut("p", modifiers: .command)
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
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            let handled = MainActor.assumeIsolated { self?.handleSpace(event) ?? false }
            return handled ? nil : event
        }
    }

    /// Space does the primary action in the main window, unless a text field has focus.
    private func handleSpace(_ event: NSEvent) -> Bool {
        guard event.keyCode == 49, !event.isARepeat,
              event.modifierFlags.intersection(.deviceIndependentFlagsMask).subtracting(.capsLock).isEmpty,
              let window = NSApp.keyWindow, window.title == "DELTA-TTS", window.attachedSheet == nil,
              !(window.firstResponder is NSText) else { return false }
        model.performPrimaryAction()
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        model.cleanUpForQuit()
    }
}
