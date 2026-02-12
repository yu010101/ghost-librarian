# Ghost Librarian TUI Demo
# Record with: vhs demo.tape

Output demo.gif
Set FontSize 14
Set Width 960
Set Height 540
Set Theme "Dracula"
Set TypingSpeed 50ms

# Start the chat TUI
Type "ghost-lib chat -m phi3:instruct"
Enter
Sleep 3s

# Type a question
Type "What is context distillation?"
Sleep 500ms
Enter

# Wait for distillation + streaming response
Sleep 12s

# Quit
Escape
Sleep 500ms
