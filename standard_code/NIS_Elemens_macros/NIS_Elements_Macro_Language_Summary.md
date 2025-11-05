# NIS Elements Macro Language Summary

This document provides a comprehensive summary of the NIS Elements macro language syntax and features based on the official Nikon documentation.

## Overview

The NIS Elements macro language is a C-like scripting language used for automating tasks and creating custom functionality within the NIS Elements imaging software. The language provides image processing capabilities with familiar C syntax.

## Official Documentation Reference

For complete and up-to-date function reference, consult the official NIS Elements documentation:
**https://www.nisoftware.net/NikonSaleApplication/Help/Docs-D/eng_d/search.html?frompage=GS_MacroLanguageDefinition.html&hash=**

This online reference contains detailed function signatures, parameters, and examples for all NIS Elements macro language functions.

## Variable Types

The macro language implements standard C data types:
- **int** - Integer values
- **double** - Double-precision floating-point numbers  
- **char** - Character data type
- **LPSTR** - String pointer type

### Important Notes:
- **Structures and Unions**: Not supported
- **Arrays**: One and two-dimensional arrays are supported
- **Pointers**: Supported through pointer operators

## Variable Scope

### Local Variables
- **CRITICAL**: Must be declared at the beginning of macro or function only
- **Cannot declare variables in the middle of functions** (unlike modern C)
- All variable declarations must appear before any executable statements
- Scope limited to the function/macro where declared

### Global Variables  
- Must be declared at the beginning of macro only
- Declared using the `global` keyword prefix
- Accessible from all function scopes within the macro interpreter
- Multiple nested macros can declare the same global variables if they are of the same type

**Example:**
```c
global int Number_Rows;
global char buffer[200];
```

## Supported Statements

The macro language supports standard C control flow statements:
- `if/else` statements
- `for` loops  
- `while` loops
- `do-while` loops
- `switch/case` statements
- `break` and `continue`
- `return` statements

## Directives

### #importUC
Used to import UnderC functions for enhanced functionality.

**Example:**
```c
#importUC DisplayCurrentPicture;

int main() {
    int rows, cols;
    AddUndoImage();
    rows = 256;
    cols = 256;
    SetCommandText("Working...");
    Get_Size(SIZE_PICTURE, NULL, NULL, &cols, &rows);
    Enable_Interrupt_InScript(2);
    inter_sharpen(cols, rows);
    DisplayCurrentPicture();
}
```

## Operators

### Arithmetic Operators
- `+` Addition
- `-` Subtraction  
- `*` Multiplication
- `/` Division
- `%` Modulo

### Assignment Operators
- `=` Simple assignment
- `+=`, `-=`, `*=`, `/=` Compound assignment operators

### Bitwise Operators
- `&` Bitwise AND
- `|` Bitwise OR
- `^` Bitwise XOR
- `~` Bitwise NOT
- `<<` Left shift
- `>>` Right shift

### Relational Operators
- `==` Equal to
- `!=` Not equal to
- `<` Less than
- `>` Greater than
- `<=` Less than or equal to
- `>=` Greater than or equal to

### Logical Operators
- `&&` Logical AND
- `||` Logical OR
- `!` Logical NOT

### Pointer Operators
- `*` Dereference operator
- `&` Address-of operator

## ⚠️ Critical Language Differences from Standard C

**NIS Elements macro language has several important differences from standard C:**

### 1. Variable Declaration Rules
- **ALL variables must be declared at the beginning of functions**
- **Cannot declare variables in the middle of code blocks**
- This is like old C89/C90 standard, not modern C99+

**Problematic Example:**
```c
int main() {
    int a = 5;
    // some code here
    int b = 10;  // ERROR: Cannot declare here
}
```

**Correct Approach:**
```c
int main() {
    int a;
    int b;
    // All declarations first
    a = 5;
    // code here
    b = 10;  // OK: Assignment, not declaration
}
```

### 2. String Literals (CONFIRMED WORKING)
- **✅ String literals with double quotes ARE SUPPORTED**
- **✅ `char *variable = "string literal";` works perfectly**
- **This greatly simplifies macro development**

**Working Examples:**
```c
char *file_path = "C:/some/path/file.py";
char *message = "Processing complete";
char *script_name = "convert_to_ometif.py";
```

**Note on Character Literals:**
- Single quotes for individual characters may still be problematic
- Use ASCII values for character comparisons: `if (ch == 47)` for '/'

### 3. Increment/Decrement Operators
- **Cannot use `++` and `--` operators**
- Must use explicit assignment form

**Problematic Example:**
```c
i++;              // ERROR: Cannot evaluate
count--;          // ERROR: Cannot evaluate
```

**Correct Approach:**
```c
i = i + 1;        // Use explicit addition
count = count - 1; // Use explicit subtraction
```

### 4. Operator Precedence
- **Logical operators are evaluated from RIGHT to LEFT**
- **Always use brackets to define evaluation order**

**Problematic Example:**
```c
if (!a && !b)  // This is evaluated as: !(a && !(b))
```

**Correct Approach:**
```c
if ((!a) && (!b))  // Use explicit brackets
```

### Standard Precedence:
- Division (`/`), multiplication (`*`), and modulo (`%`) have higher priority than other arithmetic operators
- Always use parentheses to ensure correct evaluation order

### 5. String Function Limitations
- **✅ String literals work perfectly: `char *str = "text";`**
- **NkString.dll functions like `strncopy` may cause crashes**
- **Avoid external DLL string functions when possible**
- **Use string literals for constants, manual operations for manipulation**

**✅ Recommended Approach - Use String Literals:**
```c
char *python_script = "C:/git/NIS-Elements-Python-Analysis-Servers/bf2ometif/convert_to_ometif.py";
char *completion_file = "BIPHUB_CONVERSION_COMPLETE.txt";
char *filter_string = "Image Files|*.czi;*.nd2;*.ims;*.tif;*.tiff|All Files|*.*|";
```

**For String Manipulation (when needed):**
```c
// Manual character copying when modifying strings
for (i = 0; i < length && source[i] != 0; i = i + 1) {
    dest[i] = source[i];
}
dest[i] = 0;  // Null terminate
```

### 6. sprintf Function - CRITICAL LIMITATIONS CONFIRMED
- **✅ CONFIRMED: sprintf ONLY supports single parameters**
- **❌ CONFIRMED: Multiple parameters ALWAYS FAIL with "incorrect number of parameters"**
- **❌ sprintf(buf, "%s %s", str1, str2) does NOT work**
- **✅ CONFIRMED: strcat is the reliable solution for complex strings**

**✅ sprintf - Single Parameter Only:**
```c
sprintf(buffer, "Test file: %s", "filename");     // ✅ Works
sprintf(buffer, "Processing file %d", "number");  // ✅ Works  
sprintf(buffer, "Path/%s", "script_name");        // ✅ Works
```

**❌ sprintf - Multiple Parameters FAIL:**
```c
sprintf(buffer, "%s, %s", "str1", "str2");          // ❌ FAILS: "incorrect number of parameters"
sprintf(buffer, "File %d is %s", "num", "name");    // ❌ FAILS: "incorrect number of parameters"  
sprintf(buffer, "python \"%s\" \"%s\"", "s1", "s2"); // ❌ FAILS: "incorrect number of parameters"
```

**✅ RECOMMENDED: Use strcat for Complex Strings:**
```c
// Proven reliable approach from test results:
char *python_script = "convert_to_ometif.py";
char *file_path = "test_image.czi";
char *base_path = "C:/git/NIS-Elements-Python-Analysis-Servers/bf2ometif/";

strcpy(command_buffer, "python \"");
strcat(command_buffer, base_path);
strcat(command_buffer, python_script);
strcat(command_buffer, "\" \"");
strcat(command_buffer, file_path);
strcat(command_buffer, "\"");
// Result: python "C:/git/.../convert_to_ometif.py" "test_image.czi"
```

### 7. Function Declaration Rules
- **NO forward function declarations allowed** (unlike standard C)
- **Functions must be defined before they are used**
- **Order of function definitions matters**

**Problematic Example:**
```c
// Forward declarations - NOT SUPPORTED
int ProcessFile(char* path);
int BuildPath(char* input, char* output);

int main() {
    ProcessFile("test.czi");  // ERROR: Function not yet defined
}

int ProcessFile(char* path) {
    // Function definition
}
```

**Correct Approach:**
```c
// Define utility functions first
int BuildPath(char* input, char* output) {
    // Function definition
    return 1;
}

// Define functions that use other functions second
int ProcessFile(char* path) {
    BuildPath(path, output);  // OK: BuildPath already defined
    return 1;
}

// Main function last
int main() {
    ProcessFile("test.czi");  // OK: ProcessFile already defined
}
```

## Functions

### Entry Point
- **main()** function serves as the program entry point
- If `main()` is not present, the macro body is treated as the main function (backward compatibility)
- New macros should always use `main()` as entry point

### Function Syntax
```c
int MyFunction(int a, LPSTR str, double d) {
    int retval;
    // function body
    return retval;
}
```

### Function Types
- **Interpreted functions**: C-like functions you write
- **System functions**: Built-in NIS Elements functions

## Best Practices

1. **⚠️ CRITICAL: Always declare ALL local variables at the beginning of functions/macros** (absolutely NO declarations inside code blocks!)
2. **✅ USE STRING LITERALS** - `char *variable = "string literal";` works perfectly and simplifies development
3. **Use explicit parentheses in logical expressions** due to non-standard precedence
4. **Declare global variables only at the beginning of macros**
5. **Use `main()` function as entry point for new macros**
6. **Use `Enable_Interrupt_InScript()` for long-running operations**
7. **Call `DisplayCurrentPicture()` periodically during processing for visual feedback**
8. **Avoid NkString.dll functions - use manual string operations instead**
9. **CONFIRMED: sprintf only supports single parameters** - use strcat for multiple values
10. **Test string operations thoroughly using the test macro suite**
11. **Define functions before using them - no forward declarations allowed**
12. **⚠️ CRITICAL: Avoid problematic variable names** that conflict with internal system variables

## ⚠️ CRITICAL: Reserved/Problematic Variable Names

**DISCOVERED ISSUE**: Certain variable names cause compilation failures in NIS Elements macro language. These appear to conflict with internal system variables or functions.

### Known Problematic Variables:
- `i`, `j` - Classic loop counters (may conflict with internal iterators)
- `path_len` - Path-related variables (may conflict with internal path functions)
- `pos` - Position variables (may conflict with positioning functions)  
- `result` - Result variables (may conflict with internal result storage)

### Safe Variable Naming Convention:
Use **descriptive, prefixed names** to avoid conflicts:

```c
// ❌ AVOID - May cause compilation errors:
int i, j, k;
int pos, result;
int path_len, dir_len;

// ✅ SAFE - Use descriptive names:
int idx1, idx2, loop_counter;
int buf_pos, dialog_result; 
int str_len, folder_len;
int slash_pos, dot_pos;
int file_idx, total_files;
```

### Recommended Prefixes:
- **Loop counters**: `idx1`, `idx2`, `loop_counter`
- **String operations**: `str_len`, `char_pos`, `buf_pos`
- **File operations**: `file_idx`, `total_files`, `name_start`
- **Results/Status**: `dialog_result`, `success_flag`, `error_code`
- **Positions**: `slash_pos`, `dot_pos`, `start_pos`

### ⚠️ Variable Declaration Rule Reminder:
```c
int MyFunction() {
    // ✅ ALL variables declared at the top
    int idx1, idx2, temp_pos;
    char buffer[100];
    long result_code;
    
    // ✅ Now code can begin
    if (some_condition) {
        temp_pos = buf_pos;  // ✅ Assignment OK
        // int new_var = 5;   // ❌ NEVER declare here!
    }
}
```

## Common ASCII Values for NIS Elements
Since character literals with single quotes don't work, use these numeric values:

- `0` - Null terminator (`\0`)
- `32` - Space (` `)
- `46` - Period (`.`)
- `47` - Forward slash (`/`)
- `58` - Colon (`:`)
- `92` - Backslash (`\`)
- `95` - Underscore (`_`)

**Example Usage:**
```c
if (ch == 47) {        // Check for '/'
    path[i] = 92;      // Set to '\'
}
str[len] = 0;          // Null terminate
```

## Common Functions (from examples)

- `AddUndoImage()` - Adds current state to undo history
- `SetCommandText(string)` - Sets status text
- `Get_Size()` - Gets image dimensions
- `GetPixelValue()` - Reads pixel values
- `SetPixelValue()` - Sets pixel values
- `DisplayCurrentPicture()` - Updates display
- `Enable_Interrupt_InScript()` - Controls script interruption

## Example: Image Sharpening Macro

```c
#importUC DisplayCurrentPicture;

int main() {
    int rows, cols;
    AddUndoImage();
    rows = 256;
    cols = 256;
    SetCommandText("Working...");
    Get_Size(SIZE_PICTURE, NULL, NULL, &cols, &rows);
    Enable_Interrupt_InScript(2);
    inter_sharpen(cols, rows);
    DisplayCurrentPicture();
}

__underC int inter_sharpen(int cols, int rows) {
    int i, j, value;
    for(i = 0; i < rows; i = i + 1) {
        for(j = 0; j < cols; j = j + 1) {
            value = 5 * GetPixelValue(-1, i, j, 0);
            value = value - GetPixelValue(-1, i-1, j, 0);
            value = value - GetPixelValue(-1, i+1, j, 0);  
            value = value - GetPixelValue(-1, i, j+1, 0);
            value = value - GetPixelValue(-1, i, j-1, 0);
            if(value < 0) value = 0;
            if(value > 255) value = 255;
            SetPixelValue(0, value, i, j, 0);
        }
        if(0 == i % 20)
            DisplayCurrentPicture();
    }
}
```

## Function Library Reference

Based on the macro examples, here are the most commonly used NIS Elements functions organized by category:

### Image Processing Functions

**Core Image Functions**
- `Get_ImageInfo(filename, &width, &height, NULL, &channels)` - Get image dimensions and channel count
- `Get_ImageCalibration(filename, objective, &pixel_calibration, NULL, NULL)` - Get calibration information
- `Get_Size(SIZE_PICTURE, NULL, NULL, &cols, &rows)` - Get current picture dimensions
- `GetPixelValue(layer, row, col, channel)` - Read pixel value at position
- `SetPixelValue(layer, value, row, col, channel)` - Set pixel value at position
- `DisplayCurrentPicture()` - Refresh display during processing
- `AddUndoImage()` - Add current state to undo history

**Channel Operations**
- `GetChannelName(channel_number, name_buffer, buffer_size)` - Get channel name
- `GetChannelPropertiesEx(channel, 0, &wavelength, NULL)` - Get channel properties including wavelength
- `ViewComponents(mask_string)` - Control channel visibility (e.g., "1010" for channels 1 and 3)

**Image Registration and Transformation**
- `ShiftSubPixelND(channel_name, x_shift, y_shift, ...)` - Apply sub-pixel shift to channel
- `ImageRegistration(point_count, stationary_points, moving_points, channel_mask, mode)` - Register images using control points

**Export and File Operations**
- `ImageExportAllAvailableInfo(1, filename)` - Export image metadata to file
- `ImageSaveAs(filename, format, options)` - Save image with specified format

### Binary Analysis Functions

**Layer Operations**
- `BinLayerCopy(source_layer, destination_layer)` - Copy binary layer
- `BinLayerPaste(destination_layer)` - Paste binary layer  
- `BinLayerSelect(layer_name)` - Select active binary layer
- `BinLayerDelete(pattern)` - Delete layers matching pattern (use "*" for all)
- `BinLayerDuplicate(new_name, source_layer)` - Duplicate binary layer

**Measurement and Analysis**
- `ResetRestrictions()` - Clear all measurement restrictions
- `Restrictions(parameter, enable, group, min, max)` - Set measurement restrictions
  - Example: `Restrictions("MaxFeret", 1, 1, 600, 2200)` - Filter by maximum Feret diameter
  - Example: `Restrictions("Circularity", 1, 1, 0.700, 1.000)` - Filter by circularity
  - Example: `Restrictions("MeanIntensity", 1, 1, 0.00, 80.00)` - Filter by intensity
- `GenerateBinary()` - Generate binary from current restrictions

**ROI Functions**
- `ClearMeasROI()` - Clear all measurement ROIs
- `GetROICount()` - Get number of ROIs
- `GetROIIdFromIndex(roi_index)` - Get ROI ID from index
- `GetROIInfo(id, NULL, NULL, NULL, NULL, &x, &y, NULL, NULL, NULL, NULL)` - Get ROI position and properties

### File System Functions

**Path and Directory Operations**
- `GetFileAttributesW(path)` - Get file attributes (returns `FILE_ATTRIBUTE_DOESNOTEXIST` if not found)
- `MakeDirectory(path)` - Create directory
- `ExistFile(filename)` - Check if file exists
- `DeleteFile(filename)` - Delete file
- `Get_FileSize(filename, &size)` - Get file size
- `ReadFile(filename, buffer, size)` - Read file content

**File I/O**
- `NkFile_Write(filename, text, convert_to_ascii)` - Write text to file
- `NkFile_Append(filename, text, convert_to_ascii)` - Append text to file

**System Paths**
- `GetTempPathW(buffer_size, path_buffer)` - Get temporary directory path
- `GetTempFileNameW(temp_path, prefix, unique_id, filename_buffer)` - Generate temporary filename

### User Interface Functions

**Dialog and Message Functions**
- `Int_Question(title, message, button1, button2, button3, button4, default, flags)` - Show message dialog
- `Int_GetValueEx(title, prompt, default_value, decimal_places, flags, &result)` - Get numeric input
- `SetCommandText(message)` - Set status bar text

**Window Management (NkWindow.dll)**
- `NkWindow_GetMainWindow(title, class, &handle)` - Get main window handle
- `NkWindow_FindWindow(parent, class, title, &handle)` - Find child window
- `NkWindow_SendMessage(handle, message, wparam, lparam)` - Send window message
- `NkWindow_SendMessageString(handle, message, wparam, string)` - Send string message
- `NkWindow_ShowWindow(handle, show_command)` - Show/hide window
- `NkWindow_SetWindowPos(handle, x, y)` - Set window position
- `NkWindow_GetWindowSize(handle, &width, &height)` - Get window size

**Dialog Creation (NkDialog functions)**
- `NkDialog_Create(parent_handle, title, &dialog_handle)` - Create dialog
- `NkDialog_AddControl(dialog, id, class, style, height_or_left, width, &control_handle)` - Add control
- `NkDialog_AddMacroMessageHandler(dialog, control_id, message, wparam, wparam_mask, lparam, lparam_mask, macro_code)` - Add event handler

### String Processing Functions (NkString.dll)

**String Manipulation**
- `strncopy(dest, source, count)` - Copy string with length limit
- `strncmp(str1, str2, count)` - Compare strings with length limit
- `strtok(string, delimiters, &context)` - Tokenize string
- `strsplit(offsets_array, max_count, string, tokens, flags)` - Split string into fields

**Regular Expressions**
- `RegexCompile(regex_id, pattern, flags)` - Compile regex pattern
- `RegexMatches(regex_id, string, flags)` - Test if string matches pattern
- `RegexSearch(regex_id, string, flags)` - Search for pattern in string
- `RegexGetMatch(regex_id, match_index)` - Get match result
- `RegexFree(regex_id)` - Free compiled regex

### Network Communication Functions

**Socket Operations (NkSocket.dll)**
- `NkSocket_Listen(&handle, port_or_service)` - Start listening on port
- `NkSocket_Connect(&handle, address_port_or_service)` - Connect to server
- `NkSocket_Write(handle, data, count)` - Send data
- `NkSocket_Read(handle, buffer, count, timeout)` - Receive data
- `NkSocket_WriteLine(handle, line)` - Send text line
- `NkSocket_ReadLine(handle, buffer, count, timeout)` - Receive text line
- `NkSocket_Close(&handle)` - Close connection
- `NkSocket_IsValid(handle)` - Check if socket is valid

**Important Socket Usage Notes:**
- Socket handles must be `int64` type, not `int`
- Must `import("NkSocket.dll");` before using socket functions
- Always check `NkSocket_IsValid(handle)` after connecting
- Use `NkSocket_GetErrorDescription()` for debugging connection issues
- Cast string buffers to `byte*` when needed for `NkSocket_Read`

**Example Socket Usage:**
```c
import("NkSocket.dll");
import int NkSocket_Connect(int64 * phSocket, char * addressport_or_service);
import int NkSocket_Read(int64 hSocket, byte * data, long count, long timeout);
import int NkSocket_Close(int64 * phSocket);

int main() {
    int64 socket_handle;
    char buffer[1000];
    int bytes_read;
    
    socket_handle = 0;
    if (NkSocket_Connect(&socket_handle, "127.0.0.1:50002")) {
        if (NkSocket_IsValid(socket_handle)) {
            bytes_read = NkSocket_Read(socket_handle, buffer, 999, 5000);
            if (bytes_read > 0) {
                buffer[bytes_read] = 0;  // Null terminate
                WaitText(0, buffer);
            }
        }
        NkSocket_Close(&socket_handle);
    }
}
```

**Serial Communication (NkComPort.dll)**
- `NkComPort_Open(&handle, port_and_protocol)` - Open serial port
- `NkComPort_Write(handle, buffer, count)` - Send data to port
- `NkComPort_Read(handle, buffer, count, timeout)` - Read from port
- `NkComPort_WriteLine(handle, text)` - Send text line
- `NkComPort_ReadLine(handle, buffer, count, timeout)` - Read text line
- `NkComPort_Close(&handle)` - Close port
- `NkComPort_ListPorts(buffer, buffer_size)` - List available ports

### Python Integration Functions

**Python Execution (v6_gnr_python.dll)**
- `Python_RunString(python_code)` - Execute Python code
- `Python_SetAttrInt(code, name, value)` - Set integer attribute
- `Python_SetAttrFloat(code, name, value)` - Set float attribute
- `Python_SetAttrStr(code, name, value)` - Set string attribute
- `Python_EvalInt(code, &result)` - Evaluate Python expression returning integer
- `Python_EvalFloat(code, &result)` - Evaluate Python expression returning float
- `Python_EvalStr(code, result_buffer, buffer_size)` - Evaluate Python expression returning string

### System Integration Functions

**Process Management**
- `CreateProcess(command_line, flags)` - Execute external program
- `GetCurrentProcessId()` - Get current process ID
- `WaitForSingleObject(handle, timeout_ms)` - Wait for process completion

**Memory Management** 
- `GlobalAlloc(flags, size)` - Allocate memory (use `GMEM_ZEROINIT` flag)
- `GlobalFree(memory_pointer)` - Free allocated memory
- `memset(buffer, value, size)` - Set memory block to value

**Configuration Management**
- `Get_Filename(FILE_CONFIG, path_buffer)` - Get configuration file path
- `Int_SetKeyValue(file, section, key, value)` - Write configuration value
- `Int_GetKeyValue(file, section, key, default_value)` - Read configuration value
- `Int_GetKeyString(file, section, key, buffer, buffer_size)` - Read configuration string

### Utility Functions

**Script Control**
- `Enable_Interrupt_InScript(mode)` - Allow script interruption (mode 2 recommended for long operations)
- `RunMacro(filename)` - Execute another macro file
- `ExistProc(function_name)` - Check if function/procedure exists

**Data Conversion**
- `atoi(string)` - Convert string to integer
- `atof(string)` - Convert string to double
- `sprintf(buffer, format, arguments)` - Format string (NIS-specific syntax, not standard C)

**CONFIRMED sprintf Behavior - Critical Limitations:**
```c
// ✅ String literals work perfectly:
char *script_path = "C:/git/NIS-Elements-Python-Analysis-Servers/bf2ometif/convert_to_ometif.py";
char *completion_file = "BIPHUB_CONVERSION_COMPLETE.txt";

// ✅ sprintf - ONLY single parameter works:
sprintf(buffer, "Processing %s", "filename");              // ✅ Works
sprintf(buffer, "File number %d", "42");                   // ✅ Works  
sprintf(buffer, "Path: %s", "script_path");                // ✅ Works

// ❌ sprintf - Multiple parameters ALWAYS FAIL:
sprintf(buffer, "%s %s", "str1", "str2");                  // ❌ FAILS: "incorrect number of parameters"
sprintf(buffer, "python \"%s\" \"%s\"", "s1", "s2");      // ❌ FAILS: "incorrect number of parameters"
sprintf(buffer, "%s%s%s", "prefix", "middle", "suffix");   // ❌ FAILS: "incorrect number of parameters"
```

**✅ PROVEN RELIABLE - strcat for Complex Strings (TEST CONFIRMED):**
```c
// This approach works perfectly and never crashes:
char *prefix = "python \"";
char *middle = "\" \"";  
char *suffix = "\"";

strcpy(buffer, prefix);              // Start with first part
strcat(buffer, script_path);         // Add script path
strcat(buffer, middle);              // Add middle part  
strcat(buffer, file_path);           // Add file path
strcat(buffer, suffix);              // Add final part
// Result: python "C:/path/script.py" "file.czi"
```

## Summary

The NIS Elements macro language provides a powerful C-like environment for image processing automation. Key points to remember:

- C-like syntax with some important differences in operator precedence
- Support for standard control structures and data types
- Global and local variable scoping
- Built-in image processing functions
- **Always use explicit parentheses in logical expressions**
- Use `main()` function as entry point for new macros
- Extensive function library for image analysis, UI creation, file I/O, networking, and system integration
- Support for external DLL imports to extend functionality
- Python integration capabilities for advanced processing

---
*Based on NIS Elements documentation - Copyright 1991-2025 Laboratory Imaging s.r.o.*