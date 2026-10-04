using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace ASMRDubberLauncher
{
    internal static partial class Program
    {
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
