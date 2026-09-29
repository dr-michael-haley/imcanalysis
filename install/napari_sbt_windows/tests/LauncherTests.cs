using System;
using System.IO;
using System.Security.Principal;
using NapariSBTLauncher;

internal static class LauncherTests
{
    private static int checks;
    private static string fixture;
    private static void Check(bool condition, string description)
    {
        if (!condition) throw new Exception(description);
        checks++;
        Console.WriteLine("PASS: " + description);
    }
    private static void Fails(Action action, string expected)
    {
        try { action(); }
        catch (InvalidOperationException error) { Check(error.Message.Contains(expected), expected); return; }
        throw new Exception("Expected failure: " + expected);
    }
    private static string Runtime(string parent, string name, string fake)
    {
        string root = Path.Combine(parent, name);
        Directory.CreateDirectory(Path.Combine(root, "Scripts"));
        File.Copy(Path.Combine(fake, "python.exe"), Path.Combine(root, "python.exe"));
        foreach (string dll in Directory.GetFiles(fake, "python*.dll"))
            File.Copy(dll, Path.Combine(root, Path.GetFileName(dll)));
        foreach (string dll in Directory.GetFiles(fake, "vcruntime*.dll"))
            File.Copy(dll, Path.Combine(root, Path.GetFileName(dll)));
        File.WriteAllText(Path.Combine(root, "pyvenv.cfg"),
            "home = " + fake + "\ninclude-system-site-packages = false\n");
        File.Copy(fixture, Path.Combine(root, "Scripts", "conda-unpack-script.py"));
        string module = Path.Combine(root, "SpatialBiologyToolkit", "napari_sbt");
        Directory.CreateDirectory(module);
        File.Copy(fixture, Path.Combine(module, "__main__.py"));
        File.WriteAllText(Path.Combine(root, "Scripts", "activate.bat"),
            "@echo off\r\nset \"NAPARISBT_TEST_HOOK=ran\"\r\nexit /b 0\r\n");
        return root;
    }
    private static int Main(string[] args)
    {
        try
        {
            string parent = args[0], fake = args[1], data = Path.Combine(parent, "user data");
            fixture = args[2];
            string root = Runtime(parent, "Tissue café & 20% folder", fake);
            Environment.SetEnvironmentVariable("PYTHONPATH", "must not leak");
            Environment.SetEnvironmentVariable("PYTHONHOME", "must not leak");
            var core = new LauncherCore(root, data, delegate { });
            core.Prepare();
            Check(File.ReadAllText(Path.Combine(root, ".naparisbt-location.txt")) == root, "first run records its exact location");
            core.Prepare();
            Check(File.ReadAllLines(Path.Combine(root, "calls.txt")).Length == 1, "second run skips unpacking");
            core.Launch(null);
            Check(File.ReadAllLines(Path.Combine(root, "calls.txt"))[1] == "launch", "launch receives arguments, activated environment and isolated Python paths");
            string log = File.ReadAllText(core.LogPath);
            Check(log.Contains("café") && log.Contains("Diagnostic stderr"), "stdout, stderr and Unicode are logged");
            bool admin = new WindowsPrincipal(WindowsIdentity.GetCurrent()).IsInRole(WindowsBuiltInRole.Administrator);
            Check(File.ReadAllText(Path.Combine(root, "admin.txt")) == admin.ToString(), "child uses the existing user token");
            Console.WriteLine("Test account elevated: " + admin);

            File.WriteAllText(Path.Combine(root, "fail-app"), "");
            Fails(delegate { core.Launch(null); }, "stopped with an error");
            Check(File.ReadAllText(core.LogPath).Contains("42"), "failed application exit code is logged");
            File.WriteAllText(Path.Combine(root, ".naparisbt-location.txt"), Path.Combine(parent, "old location"));
            Fails(core.Prepare, "moved or been renamed");

            string broken = Runtime(parent, "failed setup", fake);
            File.WriteAllText(Path.Combine(broken, "fail-unpack"), "");
            var failed = new LauncherCore(broken, data, delegate { });
            Fails(failed.Prepare, "could not finish its first setup");
            Check(!File.Exists(Path.Combine(broken, ".naparisbt-location.txt")), "failed setup never creates a ready marker");
            File.Delete(Path.Combine(broken, "fail-unpack"));
            failed.Prepare();
            Check(File.Exists(Path.Combine(broken, ".naparisbt-location.txt")), "failed setup can be retried");

            string old = Runtime(parent, "old launcher", fake);
            File.WriteAllText(Path.Combine(old, ".naparisbt-ready"), "");
            Fails(new LauncherCore(old, data, delegate { }).Prepare, "older launcher");
            string missing = Runtime(parent, "missing setup", fake);
            File.Delete(Path.Combine(missing, "Scripts", "conda-unpack-script.py"));
            Fails(new LauncherCore(missing, data, delegate { }).Prepare, "setup script is missing");
            Fails(new LauncherCore(parent, data, delegate { }).Prepare, "Extract the entire");
            File.WriteAllText(Path.Combine(missing, ".naparisbt-location.txt"), missing);
            File.WriteAllText(Path.Combine(missing, "Scripts", "activate.bat"), "@exit /b 9\r\n");
            Fails(delegate { new LauncherCore(missing, data, delegate { }).Launch(null); }, "could not be activated");
            Console.WriteLine(checks + " checks passed.");
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
