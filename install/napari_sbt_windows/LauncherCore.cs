using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

namespace NapariSBTLauncher
{
    // No machine-wide settings or elevation: all writes go to the runtime folder
    // during relocation, or to the current user's local application-data folder.
    internal sealed class LauncherCore
    {
        internal readonly string Root;
        internal readonly string LogPath;
        private readonly string userData;
        private readonly Action<string> status;
        private readonly object logLock = new object();

        internal LauncherCore(string root, string localData, Action<string> status)
        {
            Root = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar);
            userData = Path.Combine(localData, "NapariSBT");
            this.status = status;
            Directory.CreateDirectory(Path.Combine(userData, "logs"));
            LogPath = Path.Combine(userData, "logs", "launcher-" +
                DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff") + "-" +
                Guid.NewGuid().ToString("N").Substring(0, 8) + ".log");
            Log("Runtime: " + Root);
        }

        internal void Log(string message)
        {
            lock (logLock)
                File.AppendAllText(LogPath, DateTime.UtcNow.ToString("o") + " " +
                    message + Environment.NewLine, Encoding.UTF8);
        }

        private void LogOutput(string message)
        {
            // Pipe callbacks run on background threads. A full disk or changed
            // permissions must not crash the launcher from one of those threads.
            // The synchronous final log write still reports the error to the UI.
            try { Log(message); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }

        internal static string Quote(string argument)
        {
            // Windows CommandLineToArgvW quoting, including trailing backslashes.
            StringBuilder output = new StringBuilder("\"");
            int slashes = 0;
            foreach (char value in argument)
            {
                if (value == '\\') { slashes++; continue; }
                if (value == '"') output.Append('\\', slashes * 2 + 1);
                else output.Append('\\', slashes);
                output.Append(value);
                slashes = 0;
            }
            output.Append('\\', slashes * 2);
            return output.Append('"').ToString();
        }

        private string RuntimeId()
        {
            using (SHA256 hash = SHA256.Create())
                return BitConverter.ToString(hash.ComputeHash(
                    Encoding.UTF8.GetBytes(Root.ToUpperInvariant()))).Replace("-", "");
        }

        internal ProcessStartInfo PythonInfo(string arguments)
        {
            ProcessStartInfo info = new ProcessStartInfo(Path.Combine(Root, "python.exe"), arguments);
            info.WorkingDirectory = Root;
            info.UseShellExecute = false;
            info.CreateNoWindow = true;
            info.RedirectStandardOutput = true;
            info.RedirectStandardError = true;
            info.StandardOutputEncoding = Encoding.UTF8;
            info.StandardErrorEncoding = Encoding.UTF8;
            info.EnvironmentVariables["PATH"] = String.Join(";", new string[] {
                Root, Path.Combine(Root, "Library", "mingw-w64", "bin"),
                Path.Combine(Root, "Library", "usr", "bin"),
                Path.Combine(Root, "Library", "bin"), Path.Combine(Root, "Scripts"),
                info.EnvironmentVariables["PATH"] ?? ""
            });
            Isolate(info);
            return info;
        }

        private void Isolate(ProcessStartInfo info)
        {
            foreach (string key in new string[] { "PYTHONHOME", "PYTHONPATH", "QT_PLUGIN_PATH",
                "QML2_IMPORT_PATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL" })
                info.EnvironmentVariables.Remove(key);
            info.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
            info.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            info.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
            string cache = Path.Combine(userData, "cache", "numba", RuntimeId().Substring(0, 12));
            Directory.CreateDirectory(cache);
            info.EnvironmentVariables["NUMBA_CACHE_DIR"] = cache;
        }

        private int Run(ProcessStartInfo info, Action windowOpened)
        {
            using (Process child = new Process())
            {
                child.StartInfo = info;
                child.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) {
                    if (e.Data != null) LogOutput(e.Data);
                };
                child.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) {
                    if (e.Data != null) LogOutput("stderr: " + e.Data);
                };
                child.Start();
                child.BeginOutputReadLine();
                child.BeginErrorReadLine();
                bool shown = false;
                while (!child.WaitForExit(200))
                {
                    if (!shown && windowOpened != null)
                    {
                        child.Refresh();
                        if (child.MainWindowHandle != IntPtr.Zero)
                        {
                            shown = true;
                            windowOpened();
                        }
                    }
                }
                child.WaitForExit(); // Drain redirected output before closing the log.
                Log("Process exit code: " + child.ExitCode);
                return child.ExitCode;
            }
        }

        internal void Prepare()
        {
            if (!File.Exists(Path.Combine(Root, "python.exe")) ||
                !File.Exists(Path.Combine(Root, "Scripts", "activate.bat")))
                throw new InvalidOperationException(
                    "Extract the entire NapariSBT ZIP first. NapariSBT.exe must be beside " +
                    "python.exe, with the Lib, Library and Scripts folders alongside it.");

            using (Mutex gate = new Mutex(false, "Local\\NapariSBT-" + RuntimeId()))
            {
                bool acquired;
                try { acquired = gate.WaitOne(0); }
                catch (AbandonedMutexException) { acquired = true; }
                if (!acquired) throw new InvalidOperationException(
                    "Another NapariSBT launcher is preparing this folder. Please wait and try again.");
                try { PrepareLocked(); }
                finally { gate.ReleaseMutex(); }
            }
        }

        private void PrepareLocked()
        {
            string marker = Path.Combine(Root, ".naparisbt-location.txt");
            if (File.Exists(marker))
            {
                string previous = File.ReadAllText(marker, Encoding.UTF8).Trim();
                if (!String.Equals(previous, Root, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException(
                        "This application folder has moved or been renamed since first setup. " +
                        "Extract the original ZIP into the location you want to keep, then start " +
                        "the freshly extracted copy. Your separately saved workspaces can be reopened.");
                return;
            }
            if (File.Exists(Path.Combine(Root, ".naparisbt-ready")))
                throw new InvalidOperationException(
                    "This folder was prepared by an older launcher. Extract the original ZIP " +
                    "into a fresh folder and add NapariSBT.exe before starting it.");

            string unpack = Path.Combine(Root, "Scripts", "conda-unpack-script.py");
            if (!File.Exists(unpack)) throw new InvalidOperationException(
                "The environment setup script is missing. Extract a fresh conda-pack ZIP.");

            status("Preparing NapariSBT for this computer…\nThis only happens on the first launch.");
            string probe = Path.Combine(Root, ".naparisbt-write-" + Guid.NewGuid().ToString("N"));
            try
            {
                using (FileStream file = new FileStream(probe, FileMode.CreateNew, FileAccess.Write)) { }
                File.Delete(probe);
            }
            catch (UnauthorizedAccessException)
            {
                throw new InvalidOperationException(
                    "First setup needs a writable application folder. Extract the ZIP into " +
                    "your own local folders, such as Local AppData. Administrator access is not needed.");
            }
            if (Run(PythonInfo(Quote(unpack)), null) != 0)
                throw new InvalidOperationException(
                    "The bundled environment could not finish its first setup. The log contains " +
                    "the details. Check that the ZIP was fully extracted into a writable local folder.");
            // Never mark a partial/failed unpack as ready. Atomic rename prevents partial markers.
            File.WriteAllText(probe, Root, new UTF8Encoding(false));
            File.Move(probe, marker);
            Log("Environment relocation completed.");
        }

        private Dictionary<string, string> ActivatedEnvironment()
        {
            // Activation hooks may set data/library locations needed by scientific packages.
            // Use cmd only for the bundled activation script, not for Python arguments.
            ProcessStartInfo info = PythonInfo("");
            info.FileName = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "cmd.exe");
            info.Arguments = "/d /u /s /c \"\"%SBT_ACTIVATE_SCRIPT%\" >nul && set\"";
            info.EnvironmentVariables["SBT_ACTIVATE_SCRIPT"] = Path.Combine(Root, "Scripts", "activate.bat");
            info.StandardOutputEncoding = Encoding.Unicode;
            info.StandardErrorEncoding = Encoding.Unicode;
            using (Process process = Process.Start(info))
            {
                // Read both pipes concurrently so verbose activation failures cannot deadlock.
                var output = process.StandardOutput.ReadToEndAsync();
                var error = process.StandardError.ReadToEndAsync();
                process.WaitForExit();
                System.Threading.Tasks.Task.WaitAll(output, error);
                if (process.ExitCode != 0)
                {
                    Log("Activation: " + error.Result);
                    throw new InvalidOperationException("The bundled environment could not be activated. See the log for details.");
                }
                Dictionary<string, string> values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                foreach (string line in output.Result.Split(new char[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
                {
                    int split = line.IndexOf('=');
                    if (split > 0) values[line.Substring(0, split)] = line.Substring(split + 1);
                }
                if (!values.ContainsKey("PATH"))
                    throw new InvalidOperationException("The bundled activation script returned no environment.");
                return values;
            }
        }

        internal void Launch(Action windowOpened)
        {
            status("Starting NapariSBT…\nThe first launch may take a little longer.");
            ProcessStartInfo info = PythonInfo("-m SpatialBiologyToolkit.napari_sbt --welcome");
            foreach (var pair in ActivatedEnvironment()) info.EnvironmentVariables[pair.Key] = pair.Value;
            info.EnvironmentVariables.Remove("SBT_ACTIVATE_SCRIPT");
            info.EnvironmentVariables.Remove("PYTHONHOME");
            info.EnvironmentVariables.Remove("PYTHONPATH");
            info.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
            if (Run(info, windowOpened) != 0)
                throw new InvalidOperationException(
                    "NapariSBT stopped with an error. Open the logs folder and share the latest " +
                    "launcher log with the person who provided the application.");
        }
    }
}
