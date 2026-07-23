using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

[assembly: AssemblyTitle("ASMR Dubber")]
[assembly: AssemblyDescription("ASMR Dubber portable command-line launcher")]
[assembly: AssemblyCompany("ASMR Dubber contributors")]
[assembly: AssemblyProduct("ASMR Dubber")]
[assembly: AssemblyCopyright("Copyright (c) ASMR Dubber contributors")]
[assembly: AssemblyVersion("0.4.0.0")]
[assembly: AssemblyFileVersion("0.4.0.0")]

namespace ASMRDubberLauncher
{
    internal static class Program
    {
        private const string MutexName = "Local\\ASMRDubberPortableLauncher";
        private const string LocalUrl = "http://127.0.0.1:7860";
        private static Process activeProcess;
        private static IntPtr jobHandle;
        private static bool stopping;

        private static int Main(string[] args)
        {
            Console.OutputEncoding = new UTF8Encoding(false);
            Console.Title = "ASMR Dubber";
            string root = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);

            if (args.Length == 2 && args[0] == "--self-test")
            {
                WriteSelfTest(root, args[1]);
                return 0;
            }
            if (args.Length == 2 && args[0] == "--test-profile-prompt")
            {
                string selected = PromptForProfile();
                File.WriteAllText(args[1], selected, new UTF8Encoding(false));
                return 0;
            }

            bool created;
            using (Mutex mutex = new Mutex(true, MutexName, out created))
            {
                if (!created)
                {
                    Console.WriteLine("ASMR Dubber 已经在运行，正在打开浏览器……");
                    OpenBrowser();
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
            string setupScript = Path.Combine(root, "scripts", "windows", "setup.ps1");
            string runScript = Path.Combine(root, "scripts", "windows", "run-ui.ps1");
            if (!File.Exists(setupScript) || !File.Exists(runScript))
            {
                throw new FileNotFoundException(
                    "项目文件不完整。请保持 ASMR-Dubber.exe 位于项目根目录，并重新下载 scripts 目录。");
            }

            PrintHeader();
            if (!IsInstalled(root))
            {
                string profile = PromptForProfile();
                Console.WriteLine();
                WriteInfo("开始安装 " + profile + "。安装文件全部保存在当前项目目录。");
                Console.WriteLine("项目目录：" + root);
                Console.WriteLine();

                int setupExitCode = RunPowerShellAndWait(
                    root,
                    setupScript,
                    "-Profile " + profile);
                if (setupExitCode != 0 || !IsInstalled(root))
                {
                    throw new InvalidOperationException(
                        "安装未完成，退出码 " + setupExitCode + "。可以再次双击 EXE 续传和重试。");
                }
                Console.WriteLine();
                WriteSuccess("安装完成。");
            }

            if (ServerResponds())
            {
                WriteSuccess("检测到已经运行的 ASMR Dubber，正在打开浏览器。");
                OpenBrowser();
                return 0;
            }

            Console.WriteLine();
            WriteInfo("正在启动 ASMR Dubber……");
            Console.WriteLine("运行期间请保留此终端窗口；按 Ctrl+C 或关闭窗口即可停止。");
            Console.WriteLine();

            activeProcess = StartPowerShell(root, runScript, "");
            DateTime deadline = DateTime.UtcNow.AddSeconds(90);
            bool opened = false;
            while (!activeProcess.HasExited)
            {
                if (!opened && ServerResponds())
                {
                    opened = true;
                    WriteSuccess("服务已就绪：" + LocalUrl);
                    OpenBrowser();
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

        private static string PromptForProfile()
        {
            Console.WriteLine("首次运行，请选择安装配置：");
            Console.WriteLine();

            WriteChoice(
                "1",
                "Core · 最小安装",
                "只安装程序、网页界面和基础音频依赖，不下载大型 ASR/TTS 权重。");
            Console.WriteLine(
                "     安装完成后，需要在软件的“设置 → 设备与模型”中安装所需模型。");
            Console.WriteLine();

            WriteChoice(
                "2",
                "Recommended · 推荐安装",
                "安装 Parakeet 1.1B/0.6B、Kotoba/Faster-Whisper 运行依赖；");
            Console.WriteLine(
                "     NVIDIA 设备还会安装 IndexTTS2，约需 30 GB。其他模型仍可在软件内按需安装。");
            Console.WriteLine();

            WriteChoice(
                "3",
                "Full · 完整依赖",
                "包含 Recommended，并准备 Qwen3-ASR、ForcedAligner、VoxCPM2 权重");
            Console.WriteLine(
                "     以及更多 ASR 运行依赖。外部服务和未选择的模型仍需在软件内配置。");
            Console.WriteLine();

            while (true)
            {
                Console.Write("请输入 1、2 或 3（直接回车使用 Recommended）：");
                string input = Console.ReadLine();
                if (input == null)
                {
                    throw new InvalidOperationException("没有收到安装配置输入。");
                }
                input = input.Trim();
                if (input == "1")
                {
                    return "Core";
                }
                if (input == "" || input == "2")
                {
                    return "Recommended";
                }
                if (input == "3")
                {
                    return "Full";
                }
                WriteError("输入无效，请输入 1、2 或 3。");
            }
        }

        private static void WriteChoice(string number, string title, string description)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("  [" + number + "] " + title);
            Console.ResetColor();
            Console.WriteLine();
            Console.WriteLine("     " + description);
        }

        private static bool IsInstalled(string root)
        {
            return File.Exists(Path.Combine(
                       root, ".asmr-dubber", "venv", "Scripts", "python.exe"))
                && File.Exists(Path.Combine(
                       root, ".asmr-dubber", "venv", "Scripts", "asmr-dubber.exe"));
        }

        private static int RunPowerShellAndWait(
            string root,
            string script,
            string arguments)
        {
            activeProcess = StartPowerShell(root, script, arguments);
            activeProcess.WaitForExit();
            int exitCode = activeProcess.ExitCode;
            activeProcess = null;
            return exitCode;
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

        private static bool ServerResponds()
        {
            try
            {
                HttpWebRequest request = (HttpWebRequest)WebRequest.Create(LocalUrl + "/");
                request.Method = "GET";
                request.Timeout = 1000;
                using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                {
                    return (int)response.StatusCode >= 200
                        && (int)response.StatusCode < 500;
                }
            }
            catch
            {
                return false;
            }
        }

        private static void OpenBrowser()
        {
            try
            {
                Process.Start(LocalUrl);
            }
            catch
            {
                Console.WriteLine("请在浏览器中打开：" + LocalUrl);
            }
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
            string setup = Path.Combine(root, "scripts", "windows", "setup.ps1");
            string run = Path.Combine(root, "scripts", "windows", "run-ui.ps1");
            string python = Path.Combine(
                root, ".asmr-dubber", "venv", "Scripts", "python.exe");
            string cli = Path.Combine(
                root, ".asmr-dubber", "venv", "Scripts", "asmr-dubber.exe");
            string result = string.Join(
                Environment.NewLine,
                new[]
                {
                    "root=" + root,
                    "setup=" + File.Exists(setup),
                    "run=" + File.Exists(run),
                    "installed=" + (File.Exists(python) && File.Exists(cli)),
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
