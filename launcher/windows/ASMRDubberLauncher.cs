using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

[assembly: AssemblyTitle("ASMR Dubber")]
[assembly: AssemblyDescription("ASMR Dubber command-line launcher")]
[assembly: AssemblyCompany("ASMR Dubber contributors")]
[assembly: AssemblyProduct("ASMR Dubber")]
[assembly: AssemblyCopyright("Copyright (c) ASMR Dubber contributors")]
[assembly: AssemblyVersion("1.1.3.0")]
[assembly: AssemblyFileVersion("1.1.3.0")]

namespace ASMRDubberLauncher
{
    internal static class Program
    {
        private const string ProductMarker = "asmr-dubber-product-marker";
        private static Process activeProcess;
        private static IntPtr jobHandle;
        private static bool stopping;
        private static bool ownsPortFile;
        private static string localUrl = "http://127.0.0.1:7860";

        private static int Main(string[] args)
        {
            Console.OutputEncoding = new UTF8Encoding(false);
            Console.Title = "ASMR Dubber";
            string root = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);
            string mutexName = "Local\\ASMRDubberPortableLauncher-" + ProjectPathHash(root);

            if (args.Length == 2 && args[0] == "--self-test")
            {
                WriteSelfTest(root, args[1]);
                return 0;
            }
            bool created;
            using (Mutex mutex = new Mutex(true, mutexName, out created))
            {
                if (!created)
                {
                    Console.WriteLine("ASMR Dubber 已经在运行，正在打开浏览器……");
                    int runningPort = ReadSavedPort(root);
                    string runningUrl = UrlForPort(runningPort);
                    if (ServerResponds(runningUrl))
                    {
                        OpenBrowser(runningUrl);
                    }
                    else
                    {
                        Console.WriteLine("暂时无法确认网页端口，请稍后再运行一次启动器。");
                    }
                    return 0;
                }

                Console.CancelKeyPress += CancelRequested;
                InitializeChildProcessJob();
                try
                {
                    return Run(root);
                }
                catch (Exception exception)
                {
                    WriteError("启动失败：" + exception.Message);
                    Console.WriteLine();
                    Console.WriteLine("按任意键关闭窗口。");
                    Console.ReadKey(true);
                    return 1;
                }
                finally
                {
                    StopActiveProcessTree();
                    if (ownsPortFile)
                    {
                        DeleteSavedPort(root);
                    }
                    if (jobHandle != IntPtr.Zero)
                    {
                        NativeMethods.CloseHandle(jobHandle);
                        jobHandle = IntPtr.Zero;
                    }
                }
            }
        }

        private static int Run(string root)
        {
            string runScript = Path.Combine(root, "scripts", "windows", "run-ui.ps1");
            if (!File.Exists(runScript))
            {
                throw new FileNotFoundException(
                    "项目文件不完整。请保持 ASMR-Dubber.exe 位于项目根目录，并重新下载 scripts 目录。");
            }

            PrintHeader();
            RepairPortablePaths(root);
            if (!IsInstalled(root))
            {
                WriteError("程序依赖未安装、安装未完成或已经损坏。");
                Console.WriteLine("请运行项目根目录的 ASMR-Dubber-Setup.exe 进行安装或修复。");
                Console.WriteLine();
                Console.WriteLine("按任意键关闭窗口。");
                Console.ReadKey(true);
                return 2;
            }

            int previousPort = ReadSavedPort(root);
            string previousUrl = UrlForPort(previousPort);
            if (ServerResponds(previousUrl))
            {
                WriteSuccess("检测到已经运行的 ASMR Dubber，正在打开浏览器。");
                OpenBrowser(previousUrl);
                return 0;
            }

            int port = FindAvailablePort(7860, 100);
            localUrl = UrlForPort(port);
            SavePort(root, port);
            ownsPortFile = true;

            Console.WriteLine();
            WriteInfo("正在启动 ASMR Dubber……");
            Console.WriteLine("运行期间请保留此终端窗口；按 Ctrl+C 或关闭窗口即可停止。");
            Console.WriteLine();

            activeProcess = StartPowerShell(
                root,
                runScript,
                "-HostAddress \"127.0.0.1\" -Port " + port);
            DateTime deadline = DateTime.UtcNow.AddSeconds(90);
            bool opened = false;
            while (!activeProcess.HasExited)
            {
                if (!opened && ServerResponds(localUrl))
                {
                    opened = true;
                    WriteSuccess("服务已就绪：" + localUrl);
                    OpenBrowser(localUrl);
                }
                if (!opened && DateTime.UtcNow > deadline)
                {
                    WriteError("浏览器自动打开等待超时，请查看上方日志。");
                    deadline = DateTime.MaxValue;
                }
                Thread.Sleep(500);
            }

            int exitCode = activeProcess.ExitCode;
            activeProcess = null;
            if (stopping)
            {
                return 0;
            }
            if (exitCode != 0)
            {
                WriteError("ASMR Dubber 已退出，退出码 " + exitCode + "。");
                Console.WriteLine("按任意键关闭窗口。");
                Console.ReadKey(true);
            }
            return exitCode;
        }

        private static void PrintHeader()
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("============================================================");
            Console.WriteLine("ASMR Dubber");
            Console.WriteLine("============================================================");
            Console.ResetColor();
            Console.WriteLine();
        }

        private static bool IsInstalled(string root)
        {
            string python = Path.Combine(
                root, ".asmr-dubber", "venv", "Scripts", "python.exe");
            if (!File.Exists(python)
                || !File.Exists(Path.Combine(
                    root, ".asmr-dubber", "venv", "Scripts", "asmr-dubber.exe"))
                || !Directory.Exists(Path.Combine(
                    root, ".asmr-dubber", "venv", "Lib", "site-packages", "gradio")))
            {
                return false;
            }
            try
            {
                ProcessStartInfo info = new ProcessStartInfo();
                info.FileName = python;
                info.Arguments = "-c \"import asmr_dubber.ui, gradio, av, soundfile\"";
                info.WorkingDirectory = root;
                info.UseShellExecute = false;
                info.CreateNoWindow = true;
                using (Process check = Process.Start(info))
                {
                    if (check == null || !check.WaitForExit(30000))
                    {
                        if (check != null) check.Kill();
                        return false;
                    }
                    return check.ExitCode == 0;
                }
            }
            catch
            {
                return false;
            }
        }

        private static void RepairPortablePaths(string root)
        {
            string python = Path.Combine(
                root, ".asmr-dubber", "venv", "Scripts", "python.exe");
            if (!File.Exists(python))
            {
                return;
            }
            string runCli = Path.Combine(root, "scripts", "windows", "run-cli.ps1");
            if (!File.Exists(runCli))
            {
                return;
            }
            using (Process repair = StartPowerShell(root, runCli, "--help"))
            {
                if (!repair.WaitForExit(90000))
                {
                    repair.Kill();
                    throw new TimeoutException("项目内部可移动路径修复超时。");
                }
                if (repair.ExitCode != 0)
                {
                    throw new InvalidOperationException(
                        "项目内部可移动路径修复失败，请运行 ASMR-Dubber-Setup.exe。");
                }
            }
        }

        private static Process StartPowerShell(
            string root,
            string script,
            string extraArguments)
        {
            string powershell = FindPowerShell();
            if (string.IsNullOrEmpty(powershell))
            {
                throw new FileNotFoundException(
                    "找不到 PowerShell 7 或 Windows 自带的 PowerShell。");
            }

            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = powershell;
            startInfo.Arguments =
                "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "
                + Quote(script)
                + (string.IsNullOrEmpty(extraArguments) ? "" : " " + extraArguments);
            startInfo.WorkingDirectory = root;
            startInfo.UseShellExecute = false;
            startInfo.CreateNoWindow = false;

            Process process = Process.Start(startInfo);
            if (process == null)
            {
                throw new InvalidOperationException("无法启动内部 PowerShell 进程。");
            }
            if (jobHandle != IntPtr.Zero)
            {
                NativeMethods.AssignProcessToJobObject(jobHandle, process.Handle);
            }
            return process;
        }

        private static void InitializeChildProcessJob()
        {
            jobHandle = NativeMethods.CreateJobObject(IntPtr.Zero, null);
            if (jobHandle == IntPtr.Zero)
            {
                return;
            }

            NativeMethods.JobObjectExtendedLimitInformation information =
                new NativeMethods.JobObjectExtendedLimitInformation();
            information.BasicLimitInformation.LimitFlags =
                NativeMethods.JobObjectLimitKillOnJobClose;
            int length = Marshal.SizeOf(information);
            IntPtr pointer = Marshal.AllocHGlobal(length);
            try
            {
                Marshal.StructureToPtr(information, pointer, false);
                if (!NativeMethods.SetInformationJobObject(
                    jobHandle,
                    NativeMethods.JobObjectInfoClass.ExtendedLimitInformation,
                    pointer,
                    (uint)length))
                {
                    NativeMethods.CloseHandle(jobHandle);
                    jobHandle = IntPtr.Zero;
                }
            }
            finally
            {
                Marshal.FreeHGlobal(pointer);
            }
        }

        internal static string FindPowerShell()
        {
            string programFiles = Environment.GetFolderPath(
                Environment.SpecialFolder.ProgramFiles);
            string pwsh = Path.Combine(programFiles, "PowerShell", "7", "pwsh.exe");
            if (File.Exists(pwsh))
            {
                return pwsh;
            }

            string system = Environment.GetFolderPath(
                Environment.SpecialFolder.System);
            string windowsPowerShell = Path.Combine(
                system, "WindowsPowerShell", "v1.0", "powershell.exe");
            return File.Exists(windowsPowerShell) ? windowsPowerShell : null;
        }

        private static bool ServerResponds(string url)
        {
            try
            {
                HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url + "/");
                request.Method = "GET";
                request.Timeout = 1000;
                request.ReadWriteTimeout = 1000;
                request.AutomaticDecompression =
                    DecompressionMethods.GZip | DecompressionMethods.Deflate;
                using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                {
                    if ((int)response.StatusCode < 200 || (int)response.StatusCode >= 400)
                    {
                        return false;
                    }
                    using (StreamReader reader = new StreamReader(
                        response.GetResponseStream(), Encoding.UTF8, true))
                    {
                        string body = reader.ReadToEnd();
                        return body.IndexOf(ProductMarker, StringComparison.Ordinal) >= 0;
                    }
                }
            }
            catch
            {
                return false;
            }
        }

        private static void OpenBrowser(string url)
        {
            try
            {
                Process.Start(url);
            }
            catch
            {
                Console.WriteLine("请在浏览器中打开：" + url);
            }
        }

        private static string ProjectPathHash(string root)
        {
            string normalized = root.TrimEnd(Path.DirectorySeparatorChar)
                .ToUpperInvariant();
            using (SHA256 algorithm = SHA256.Create())
            {
                byte[] digest = algorithm.ComputeHash(Encoding.UTF8.GetBytes(normalized));
                StringBuilder result = new StringBuilder(16);
                for (int index = 0; index < 8; index++)
                {
                    result.Append(digest[index].ToString("x2"));
                }
                return result.ToString();
            }
        }

        private static string PortFile(string root)
        {
            return Path.Combine(root, ".asmr-dubber", "runtime", "ui-port.txt");
        }

        private static int ReadSavedPort(string root)
        {
            try
            {
                int port;
                if (int.TryParse(File.ReadAllText(PortFile(root)).Trim(), out port)
                    && port > 0 && port <= 65535)
                {
                    return port;
                }
            }
            catch
            {
                // The first launch has no port marker yet.
            }
            return 7860;
        }

        private static void SavePort(string root, int port)
        {
            string path = PortFile(root);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, port.ToString(), new UTF8Encoding(false));
        }

        private static void DeleteSavedPort(string root)
        {
            try
            {
                File.Delete(PortFile(root));
            }
            catch
            {
                // A stale marker is harmless because every reuse verifies the page marker.
            }
        }

        private static string UrlForPort(int port)
        {
            return "http://127.0.0.1:" + port;
        }

        private static int FindAvailablePort(int first, int count)
        {
            for (int port = first; port < first + count && port <= 65535; port++)
            {
                TcpListener listener = null;
                try
                {
                    listener = new TcpListener(IPAddress.Loopback, port);
                    listener.Start();
                    return port;
                }
                catch (SocketException)
                {
                    // Try the next port.
                }
                finally
                {
                    if (listener != null)
                    {
                        listener.Stop();
                    }
                }
            }
            throw new InvalidOperationException("找不到可用的本机网页端口（7860–7959）。");
        }

        private static void CancelRequested(object sender, ConsoleCancelEventArgs e)
        {
            e.Cancel = true;
            stopping = true;
            Console.WriteLine();
            WriteInfo("正在停止 ASMR Dubber……");
            StopActiveProcessTree();
        }

        private static void StopActiveProcessTree()
        {
            Process process = activeProcess;
            if (process == null)
            {
                return;
            }
            try
            {
                if (process.HasExited)
                {
                    return;
                }
                ProcessStartInfo stopInfo = new ProcessStartInfo();
                stopInfo.FileName = "taskkill.exe";
                stopInfo.Arguments = "/PID " + process.Id + " /T /F";
                stopInfo.UseShellExecute = false;
                stopInfo.CreateNoWindow = true;
                using (Process stop = Process.Start(stopInfo))
                {
                    if (stop != null)
                    {
                        stop.WaitForExit(5000);
                    }
                }
            }
            catch
            {
                try
                {
                    process.Kill();
                }
                catch
                {
                    // The process may already have exited between checks.
                }
            }
        }

        private static string Quote(string value)
        {
            return "\"" + value.Replace("\"", "\\\"") + "\"";
        }

        private static void WriteInfo(string message)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(message);
            Console.ResetColor();
        }

        private static void WriteSuccess(string message)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(message);
            Console.ResetColor();
        }

        private static void WriteError(string message)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Error.WriteLine(message);
            Console.ResetColor();
        }

        private static void WriteSelfTest(string root, string destination)
        {
            string run = Path.Combine(root, "scripts", "windows", "run-ui.ps1");
            string installer = Path.Combine(root, "ASMR-Dubber-Setup.exe");
            string result = string.Join(
                Environment.NewLine,
                new[]
                {
                    "root=" + root,
                    "run=" + File.Exists(run),
                    "setup_exe=" + File.Exists(installer),
                    "installed=" + IsInstalled(root),
                    "powershell=" + (FindPowerShell() ?? ""),
                });
            File.WriteAllText(destination, result, new UTF8Encoding(false));
        }

        private static class NativeMethods
        {
            internal const uint JobObjectLimitKillOnJobClose = 0x00002000;

            internal enum JobObjectInfoClass
            {
                ExtendedLimitInformation = 9,
            }

            [StructLayout(LayoutKind.Sequential)]
            internal struct IoCounters
            {
                internal ulong ReadOperationCount;
                internal ulong WriteOperationCount;
                internal ulong OtherOperationCount;
                internal ulong ReadTransferCount;
                internal ulong WriteTransferCount;
                internal ulong OtherTransferCount;
            }

            [StructLayout(LayoutKind.Sequential)]
            internal struct BasicLimitInformation
            {
                internal long PerProcessUserTimeLimit;
                internal long PerJobUserTimeLimit;
                internal uint LimitFlags;
                internal UIntPtr MinimumWorkingSetSize;
                internal UIntPtr MaximumWorkingSetSize;
                internal uint ActiveProcessLimit;
                internal IntPtr Affinity;
                internal uint PriorityClass;
                internal uint SchedulingClass;
            }

            [StructLayout(LayoutKind.Sequential)]
            internal struct JobObjectExtendedLimitInformation
            {
                internal BasicLimitInformation BasicLimitInformation;
                internal IoCounters IoInfo;
                internal UIntPtr ProcessMemoryLimit;
                internal UIntPtr JobMemoryLimit;
                internal UIntPtr PeakProcessMemoryUsed;
                internal UIntPtr PeakJobMemoryUsed;
            }

            [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
            internal static extern IntPtr CreateJobObject(
                IntPtr jobAttributes,
                string name);

            [DllImport("kernel32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            internal static extern bool SetInformationJobObject(
                IntPtr job,
                JobObjectInfoClass informationClass,
                IntPtr information,
                uint informationLength);

            [DllImport("kernel32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            internal static extern bool AssignProcessToJobObject(
                IntPtr job,
                IntPtr process);

            [DllImport("kernel32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            internal static extern bool CloseHandle(IntPtr handle);
        }
    }
}
