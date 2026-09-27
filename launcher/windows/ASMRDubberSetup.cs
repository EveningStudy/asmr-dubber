using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;

[assembly: System.Reflection.AssemblyTitle("ASMR Dubber Setup")]
[assembly: System.Reflection.AssemblyDescription("ASMR Dubber dependency installer and repair tool")]
[assembly: System.Reflection.AssemblyCompany("ASMR Dubber contributors")]
[assembly: System.Reflection.AssemblyProduct("ASMR Dubber")]
[assembly: System.Reflection.AssemblyVersion("1.6.0.0")]
[assembly: System.Reflection.AssemblyFileVersion("1.6.0.0")]

namespace ASMRDubberSetup
{
    internal static class Program
    {
        private static readonly object OutputLock = new object();
        private static TextWriter originalOutput;
        private static TextWriter originalError;
        private static StreamWriter logWriter;
        private static string logPath;
        private static bool english;
        private static string T(string zh, string en) { return english ? en : zh; }

        private static int Main(string[] args)
        {
            Console.OutputEncoding = new UTF8Encoding(false);
            Console.Title = "ASMR Dubber Setup";
            string root = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);

            if (args.Length == 2 && args[0] == "--self-test")
            {
                WriteSelfTest(root, args[1]);
                return 0;
            }
            if (args.Length == 2 && args[0] == "--self-test-log")
            {
                InitializeLogging(root);
                Console.WriteLine("ASMR Dubber setup log self-test");
                string createdLog = logPath ?? "";
                CloseLogging();
                File.WriteAllText(args[1], createdLog, new UTF8Encoding(false));
                return string.IsNullOrEmpty(createdLog) ? 1 : 0;
            }
            if (args.Length == 2 && args[0] == "--test-profile-prompt")
            {
                File.WriteAllText(args[1], PromptForProfile(), new UTF8Encoding(false));
                return 0;
            }

            english = Array.IndexOf(args, "--lang=en") >= 0;
            if (Array.IndexOf(args, "--lang=en") < 0 && Array.IndexOf(args, "--lang=zh") < 0)
            {
                Console.Write("Language / 语言 [1 中文 (默认), 2 English]: ");
                english = Console.ReadLine() == "2";
            }
            Environment.SetEnvironmentVariable("ASMR_DUBBER_UI_LANGUAGE", english ? "en" : "zh");
            InitializeLogging(root);
            try
            {
                if (!string.IsNullOrEmpty(logPath))
                {
                    Console.WriteLine(T("安装日志：", "Setup log: ") + logPath);
                    Console.WriteLine();
                }
                return Run(root);
            }
            catch (Exception exception)
            {
                WriteError(T("安装或修复失败：", "Installation or repair failed: ") + exception.Message);
                Console.WriteLine();
                Console.WriteLine(T("可以再次运行 ASMR-Dubber-Setup.exe 继续下载和修复。", "Run ASMR-Dubber-Setup.exe again to resume downloads and repair."));
                WaitForClose();
                return 1;
            }
            finally
            {
                CloseLogging();
            }
        }

        private static int Run(string root)
        {
            string setupScript = Path.Combine(root, "scripts", "windows", "setup.ps1");
            string mirrors = Path.Combine(root, "mirrors.json");
            if (!File.Exists(setupScript) || !File.Exists(mirrors))
            {
                throw new FileNotFoundException(
                    T("项目文件不完整，请确认 scripts 目录和 mirrors.json 位于项目根目录。", "Incomplete application files: scripts and mirrors.json must be in the application directory."));
            }

            Console.WriteLine(T("ASMR Dubber 依赖安装与修复", "ASMR Dubber dependency installation & repair"));
            Console.WriteLine();
            Console.WriteLine(
                T("基础环境：", "Core runtime: ") + (HasCoreFiles(root) ? T("已安装", "Installed") : T("未安装或不完整", "Missing or incomplete")));
            Console.WriteLine(
                "ASR · Parakeet: "
                + (IsParakeetInstalled(root) ? T("已安装", "Installed") : T("未安装或不完整", "Missing or incomplete")));
            Console.WriteLine(
                "TTS · IndexTTS2: "
                + (IsIndexTtsInstalled(root) ? T("已安装", "Installed") : T("未安装或不完整", "Missing or incomplete")));
            Console.WriteLine(T("重复运行会复用已完成的文件，并继续未完成的下载。", "Reruns reuse completed files and resume unfinished downloads."));
            Console.WriteLine(T("默认优先使用 ModelScope；不会自动切换到 GitHub 或 Hugging Face。", "ModelScope is preferred. No automatic fallback to GitHub or Hugging Face."));
            Console.WriteLine();

            string profile = PromptForProfile();
            Console.WriteLine();
            Console.WriteLine(T("开始安装或修复 ", "Installing / repairing profile: ") + profile);
            int exitCode = RunPowerShell(root, setupScript, "-Profile " + profile);
            if (exitCode != 0)
            {
                throw new InvalidOperationException(T("安装脚本退出码 ", "Setup script exit code: ") + exitCode);
            }
            if (!HasCoreFiles(root))
            {
                throw new InvalidOperationException(
                    T("安装脚本已完成，但基础环境文件仍不完整。请保留本窗口和安装日志后重新运行 Setup。", "Setup finished but core files are incomplete. Keep this window and log, then rerun Setup."));
            }

            string coreDiagnostic;
            if (!CanImportCore(root, out coreDiagnostic))
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine();
                Console.WriteLine(T("安装脚本已成功完成，但启动器的附加导入检查未通过：", "Setup succeeded, but the additional import check failed: ") + coreDiagnostic);
                Console.WriteLine(
                    T("这项附加检查不会把成功安装改判为失败；启动程序时仍会执行正式环境检查。", "This supplementary check does not invalidate installation. Startup performs the full runtime check."));
                Console.ResetColor();
            }

            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(T("安装或修复完成。现在可以运行 ASMR-Dubber.exe。", "Installation / repair complete. Run ASMR-Dubber.exe."));
            Console.ResetColor();
            WaitForClose();
            return 0;
        }

        private static string PromptForProfile()
        {
            if (english)
            {
                Console.WriteLine("1  Core: application and web UI; no large models. ~2 GB installed; reserve 5 GB.");
                Console.WriteLine("2  Recommended: Core + Parakeet 1.1B/0.6B; IndexTTS2 on NVIDIA GPUs.");
                Console.WriteLine("   ~24-28 GB installed; reserve 35 GB.");
                Console.WriteLine("3  Advanced: Parakeet CTC 1.1B JA GAL, Parakeet TDT/CTC 0.6B JA,");
                Console.WriteLine("   Kotoba-Whisper v2.2, Faster-Whisper large-v2, Japanese ASMR VAD,");
                Console.WriteLine("   Qwen3 ForcedAligner 0.6B and IndexTTS2 (NVIDIA only).");
                Console.WriteLine("   ~33-39 GB installed; reserve 50 GB. Less space without IndexTTS2.");
                Console.WriteLine("   Does not install Kotoba v2.0/v2.1, large-v3 or IndexTTS-2.5.");
            }
            else
            {
            Console.WriteLine("1  基础：程序和网页界面，不下载大型模型");
            Console.WriteLine("   安装后约 2 GB；建议至少预留 5 GB");
            Console.WriteLine("2  推荐：基础环境、Parakeet 1.1B/0.6B；");
            Console.WriteLine("   NVIDIA 设备另外安装 TTS（语音合成）IndexTTS2");
            Console.WriteLine("   安装后约 24–28 GB；建议至少预留 35 GB");
            Console.WriteLine("3  进阶：明确安装以下 7 个固定模型");
            Console.WriteLine("   ASR（语音识别）：Parakeet CTC 1.1B JA GAL");
            Console.WriteLine("   ASR（语音识别）：Parakeet TDT/CTC 0.6B JA");
            Console.WriteLine("   ASR（语音识别）：Kotoba-Whisper v2.2");
            Console.WriteLine("   ASR（语音识别）：Faster-Whisper large-v2");
            Console.WriteLine("   VAD（语音活动检测）：日语 ASMR 专用 Whisper VAD ONNX");
            Console.WriteLine("   时间戳对齐：Qwen3 ForcedAligner 0.6B（阿里 Qwen）");
            Console.WriteLine("   TTS（语音合成）：IndexTTS2 checkpoints（仅 NVIDIA GPU）");
            Console.WriteLine("   不会自动安装 Kotoba v2.0/v2.1、large-v3 或其它识别模型");
            Console.WriteLine("   安装后约 33–39 GB；建议至少预留 50 GB");
            Console.WriteLine("   无 NVIDIA GPU 时会跳过 IndexTTS2，实际占用将减少");
            }
            Console.WriteLine();
            while (true)
            {
                Console.Write(T("选择 1、2 或 3（直接回车选择“推荐”）：", "Choose 1, 2 or 3 [Enter = Recommended; L = switch language]: "));
                string input = Console.ReadLine();
                if (input == null)
                {
                    throw new InvalidOperationException(T("没有收到输入。", "No input received."));
                }
                input = input.Trim();
                if (input == "1") return "基础";
                if (input == "" || input == "2") return "推荐";
                if (input == "3") return "进阶";
                if (string.Equals(input, "L", StringComparison.OrdinalIgnoreCase))
                {
                    english = !english;
                    Environment.SetEnvironmentVariable("ASMR_DUBBER_UI_LANGUAGE", english ? "en" : "zh");
                    return PromptForProfile();
                }
                WriteError(T("请输入 1、2 或 3（L 切换语言）。", "Enter 1, 2 or 3 (L switches language)."));
            }
        }

        private static bool HasCoreFiles(string root)
        {
            string python = Path.Combine(
                root, ".asmr-dubber", "venv", "Scripts", "python.exe");
            return File.Exists(python)
                && File.Exists(Path.Combine(
                    root, ".asmr-dubber", "venv", "Scripts", "asmr-dubber.exe"))
                && Directory.Exists(Path.Combine(
                    root, ".asmr-dubber", "venv", "Lib", "site-packages", "gradio"));
        }

        private static bool CanImportCore(string root, out string diagnostic)
        {
            string python = Path.Combine(
                root, ".asmr-dubber", "venv", "Scripts", "python.exe");
            if (!HasCoreFiles(root))
            {
                diagnostic = T("基础环境文件缺失", "Core runtime files are missing");
                return false;
            }
            try
            {
                ProcessStartInfo info = new ProcessStartInfo();
                info.FileName = python;
                info.Arguments = "-c \"import asmr_dubber, gradio, av, soundfile\"";
                info.WorkingDirectory = root;
                info.UseShellExecute = false;
                info.CreateNoWindow = true;
                using (Process check = Process.Start(info))
                {
                    if (check == null)
                    {
                        diagnostic = T("无法启动 Python 导入检查", "Could not start Python import check");
                        return false;
                    }
                    if (!check.WaitForExit(120000))
                    {
                        check.Kill();
                        diagnostic = T("Python 导入检查超过 120 秒", "Python import check timed out after 120 s");
                        return false;
                    }
                    if (check.ExitCode != 0)
                    {
                        diagnostic = T("Python 导入检查退出码 ", "Python import check exit code: ") + check.ExitCode;
                        return false;
                    }
                    diagnostic = T("检查通过", "Check passed");
                    return true;
                }
            }
            catch (Exception exception)
            {
                diagnostic = exception.Message;
                return false;
            }
        }

        private static bool IsParakeetInstalled(string root)
        {
            string home = Path.Combine(root, ".asmr-dubber");
            return File.Exists(Path.Combine(home, "runtimes", "crispasr", "bin", "crispasr.exe"))
                && File.Exists(Path.Combine(
                    home, "models", "parakeet", "parakeet-ctc-1.1b-ja-f16.gguf"))
                && File.Exists(Path.Combine(
                    home, "models", "parakeet", "parakeet-tdt-0.6b-ja.gguf"));
        }

        private static bool IsIndexTtsInstalled(string root)
        {
            string runtime = Path.Combine(root, ".asmr-dubber", "runtimes", "index-tts");
            return File.Exists(Path.Combine(runtime, ".venv", "Scripts", "indextts2.exe"))
                && File.Exists(Path.Combine(runtime, "checkpoints", "config.yaml"))
                && File.Exists(Path.Combine(runtime, "checkpoints", "gpt.pth"));
        }

        private static int RunPowerShell(string root, string script, string arguments)
        {
            string powershell = FindPowerShell();
            if (string.IsNullOrEmpty(powershell))
            {
                throw new FileNotFoundException(T("找不到 PowerShell 7 或 Windows PowerShell。", "PowerShell 7 or Windows PowerShell was not found."));
            }
            ProcessStartInfo info = new ProcessStartInfo();
            info.FileName = powershell;
            info.Arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "
                + Quote(script) + " " + arguments;
            info.WorkingDirectory = root;
            info.UseShellExecute = false;
            info.RedirectStandardOutput = true;
            info.RedirectStandardError = true;
            info.StandardOutputEncoding = new UTF8Encoding(false);
            info.StandardErrorEncoding = new UTF8Encoding(false);
            using (Process process = Process.Start(info))
            {
                if (process == null)
                {
                    throw new InvalidOperationException(T("无法启动安装脚本。", "Could not start the setup script."));
                }
                Thread outputThread = StartCopyThread(process.StandardOutput, Console.Out);
                Thread errorThread = StartCopyThread(process.StandardError, Console.Error);
                process.WaitForExit();
                outputThread.Join();
                errorThread.Join();
                return process.ExitCode;
            }
        }

        private static Thread StartCopyThread(TextReader source, TextWriter destination)
        {
            Thread thread = new Thread(delegate()
            {
                char[] buffer = new char[4096];
                int count;
                while ((count = source.Read(buffer, 0, buffer.Length)) > 0)
                {
                    destination.Write(buffer, 0, count);
                    destination.Flush();
                }
            });
            thread.IsBackground = true;
            thread.Start();
            return thread;
        }

        private static void InitializeLogging(string root)
        {
            originalOutput = Console.Out;
            originalError = Console.Error;
            try
            {
                string logDirectory = Path.Combine(root, ".asmr-dubber", "logs");
                Directory.CreateDirectory(logDirectory);
                logPath = Path.Combine(
                    logDirectory,
                    "setup-" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff") + ".log");
                logWriter = new StreamWriter(logPath, false, new UTF8Encoding(false));
                logWriter.AutoFlush = true;
                Console.SetOut(TextWriter.Synchronized(
                    new TeeTextWriter(originalOutput, logWriter)));
                Console.SetError(TextWriter.Synchronized(
                    new TeeTextWriter(originalError, logWriter)));
            }
            catch (Exception exception)
            {
                logPath = null;
                if (logWriter != null)
                {
                    logWriter.Dispose();
                    logWriter = null;
                }
                originalError.WriteLine(T("无法创建安装日志：", "Could not create setup log: ") + exception.Message);
            }
        }

        private static void CloseLogging()
        {
            if (originalOutput != null) Console.SetOut(originalOutput);
            if (originalError != null) Console.SetError(originalError);
            if (logWriter != null)
            {
                logWriter.Flush();
                logWriter.Dispose();
                logWriter = null;
            }
        }

        private static string FindPowerShell()
        {
            string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            string pwsh = Path.Combine(programFiles, "PowerShell", "7", "pwsh.exe");
            if (File.Exists(pwsh)) return pwsh;
            string system = Environment.GetFolderPath(Environment.SpecialFolder.System);
            string windowsPowerShell = Path.Combine(
                system, "WindowsPowerShell", "v1.0", "powershell.exe");
            return File.Exists(windowsPowerShell) ? windowsPowerShell : null;
        }

        private static string Quote(string value)
        {
            return "\"" + value.Replace("\"", "\\\"") + "\"";
        }

        private static void WriteError(string message)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Error.WriteLine(message);
            Console.ResetColor();
        }

        private static void WaitForClose()
        {
            Console.WriteLine(T("按任意键关闭窗口。", "Press any key to close."));
            Console.ReadKey(true);
        }

        private static void WriteSelfTest(string root, string destination)
        {
            string result = string.Join(
                Environment.NewLine,
                new[]
                {
                    "root=" + root,
                    "setup=" + File.Exists(Path.Combine(root, "scripts", "windows", "setup.ps1")),
                    "mirrors=" + File.Exists(Path.Combine(root, "mirrors.json")),
                    "installed=" + HasCoreFiles(root),
                    "powershell=" + (FindPowerShell() ?? ""),
                });
            File.WriteAllText(destination, result, new UTF8Encoding(false));
        }

        private sealed class TeeTextWriter : TextWriter
        {
            private readonly TextWriter first;
            private readonly TextWriter second;

            internal TeeTextWriter(TextWriter firstWriter, TextWriter secondWriter)
            {
                first = firstWriter;
                second = secondWriter;
            }

            public override Encoding Encoding
            {
                get { return first.Encoding; }
            }

            public override void Write(char value)
            {
                lock (OutputLock)
                {
                    first.Write(value);
                    second.Write(value);
                }
            }

            public override void Write(char[] buffer, int index, int count)
            {
                lock (OutputLock)
                {
                    first.Write(buffer, index, count);
                    second.Write(buffer, index, count);
                }
            }

            public override void Write(string value)
            {
                lock (OutputLock)
                {
                    first.Write(value);
                    second.Write(value);
                }
            }

            public override void Flush()
            {
                lock (OutputLock)
                {
                    first.Flush();
                    second.Flush();
                }
            }
        }
    }
}
