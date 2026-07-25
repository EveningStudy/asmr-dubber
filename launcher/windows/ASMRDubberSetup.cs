using System;
using System.Diagnostics;
using System.IO;
using System.Text;

[assembly: System.Reflection.AssemblyTitle("ASMR Dubber Setup")]
[assembly: System.Reflection.AssemblyDescription("ASMR Dubber dependency installer and repair tool")]
[assembly: System.Reflection.AssemblyCompany("ASMR Dubber contributors")]
[assembly: System.Reflection.AssemblyProduct("ASMR Dubber")]
[assembly: System.Reflection.AssemblyVersion("0.3.1.0")]
[assembly: System.Reflection.AssemblyFileVersion("0.3.1.0")]

namespace ASMRDubberSetup
{
    internal static class Program
    {
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
            if (args.Length == 2 && args[0] == "--test-profile-prompt")
            {
                File.WriteAllText(args[1], PromptForProfile(), new UTF8Encoding(false));
                return 0;
            }

            try
            {
                return Run(root);
            }
            catch (Exception exception)
            {
                WriteError("安装或修复失败：" + exception.Message);
                Console.WriteLine();
                Console.WriteLine("可以再次运行 ASMR-Dubber-Setup.exe 继续下载和修复。");
                WaitForClose();
                return 1;
            }
        }

        private static int Run(string root)
        {
            string setupScript = Path.Combine(root, "scripts", "windows", "setup.ps1");
            string mirrors = Path.Combine(root, "mirrors.json");
            if (!File.Exists(setupScript) || !File.Exists(mirrors))
            {
                throw new FileNotFoundException(
                    "项目文件不完整，请确认 scripts 目录和 mirrors.json 位于项目根目录。");
            }

            Console.WriteLine("ASMR Dubber 依赖安装与修复");
            Console.WriteLine();
            Console.WriteLine(
                "Core：" + (IsCoreInstalled(root) ? "已安装" : "未安装或不完整"));
            Console.WriteLine(
                "Parakeet：" + (IsParakeetInstalled(root) ? "已安装" : "未安装或不完整"));
            Console.WriteLine(
                "IndexTTS2：" + (IsIndexTtsInstalled(root) ? "已安装" : "未安装或不完整"));
            Console.WriteLine("重复运行会复用已完成的文件，并继续未完成的下载。");
            Console.WriteLine();

            string profile = PromptForProfile();
            Console.WriteLine();
            Console.WriteLine("开始安装或修复 " + profile + "。");
            int exitCode = RunPowerShell(root, setupScript, "-Profile " + profile);
            if (exitCode != 0 || !IsCoreInstalled(root))
            {
                throw new InvalidOperationException("安装脚本退出码 " + exitCode + "。");
            }

            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("安装或修复完成。现在可以运行 ASMR-Dubber.exe。");
            Console.ResetColor();
            WaitForClose();
            return 0;
        }

        private static string PromptForProfile()
        {
            Console.WriteLine("1  Core：程序和网页界面，不下载大型模型");
            Console.WriteLine("   安装后约 2 GB；建议至少预留 5 GB");
            Console.WriteLine("2  Recommended：Core、Parakeet 1.1B/0.6B；");
            Console.WriteLine("   NVIDIA 设备另外安装 IndexTTS2");
            Console.WriteLine("   安装后约 24–28 GB；建议至少预留 35 GB");
            Console.WriteLine("3  Advanced：Recommended、Kotoba-Whisper v2.2、Faster-Whisper large-v2");
            Console.WriteLine("   安装后约 30–35 GB；建议至少预留 45 GB");
            Console.WriteLine("4  Full：Advanced，加上其余已集成且可自动安装的本地后端");
            Console.WriteLine("   安装后约 42–48 GB；建议至少预留 60 GB");
            Console.WriteLine("   无 NVIDIA GPU 时会跳过 CUDA 后端，实际占用将减少");
            Console.WriteLine();
            while (true)
            {
                Console.Write("选择 1、2、3 或 4（直接回车选择 Recommended）：");
                string input = Console.ReadLine();
                if (input == null)
                {
                    throw new InvalidOperationException("没有收到输入。");
                }
                input = input.Trim();
                if (input == "1") return "Core";
                if (input == "" || input == "2") return "Recommended";
                if (input == "3") return "Advanced";
                if (input == "4") return "Full";
                WriteError("请输入 1、2、3 或 4。");
            }
        }

        private static bool IsCoreInstalled(string root)
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
                throw new FileNotFoundException("找不到 PowerShell 7 或 Windows PowerShell。");
            }
            ProcessStartInfo info = new ProcessStartInfo();
            info.FileName = powershell;
            info.Arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "
                + Quote(script) + " " + arguments;
            info.WorkingDirectory = root;
            info.UseShellExecute = false;
            using (Process process = Process.Start(info))
            {
                if (process == null)
                {
                    throw new InvalidOperationException("无法启动安装脚本。");
                }
                process.WaitForExit();
                return process.ExitCode;
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
            Console.WriteLine("按任意键关闭窗口。");
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
                    "installed=" + IsCoreInstalled(root),
                    "powershell=" + (FindPowerShell() ?? ""),
                });
            File.WriteAllText(destination, result, new UTF8Encoding(false));
        }
    }
}
