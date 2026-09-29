using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("NapariSBT")]
[assembly: System.Reflection.AssemblyDescription("Per-user launcher for packed NapariSBT environments")]
[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]

namespace NapariSBTLauncher
{
    internal sealed class StartupWindow : Form
    {
        private readonly Label message;
        private readonly ProgressBar progress;
        private readonly FlowLayoutPanel actions;
        private LauncherCore core;

        internal StartupWindow()
        {
            Text = "NapariSBT";
            Icon = SystemIcons.Application;
            StartPosition = FormStartPosition.CenterScreen;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            ControlBox = false;
            AutoScaleMode = AutoScaleMode.Dpi;
            ClientSize = new Size(480, 250);
            Font = new Font("Segoe UI", 10);
            BackColor = Color.FromArgb(245, 247, 250);
            var layout = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(24), RowCount = 4, ColumnCount = 1 };
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 18));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            layout.Controls.Add(new Label { Text = "NapariSBT", AutoSize = true,
                Font = new Font("Segoe UI", 19, FontStyle.Bold), ForeColor = Color.FromArgb(0, 112, 92) }, 0, 0);
            message = new Label { Text = "Checking the application folder…", Dock = DockStyle.Fill };
            layout.Controls.Add(message, 0, 1);
            progress = new ProgressBar { Dock = DockStyle.Fill, Style = ProgressBarStyle.Marquee };
            layout.Controls.Add(progress, 0, 2);
            actions = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.RightToLeft };
            layout.Controls.Add(actions, 0, 3);
            Controls.Add(layout);
            Shown += Start;
        }

        private void UpdateMessage(string text)
        {
            BeginInvoke((Action)delegate { message.Text = text; });
        }

        private async void Start(object sender, EventArgs args)
        {
            try
            {
                core = new LauncherCore(AppDomain.CurrentDomain.BaseDirectory,
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), UpdateMessage);
                await Task.Run(delegate {
                    core.Prepare();
                    core.Launch(delegate { BeginInvoke((Action)delegate { Hide(); }); });
                });
                Close();
            }
            catch (Exception error)
            {
                if (core != null)
                {
                    try { core.Log(error.ToString()); }
                    catch (IOException) { }
                    catch (UnauthorizedAccessException) { }
                }
                progress.Visible = false;
                ClientSize = new Size(520, 330);
                message.Text = error.Message + (core == null ? "" : "\n\nLog: " + core.LogPath);
                var close = new Button { Text = "Close", AutoSize = true };
                close.Click += delegate { Close(); };
                actions.Controls.Add(close);
                if (core != null)
                {
                    var logs = new Button { Text = "Open logs folder", AutoSize = true };
                    logs.Click += delegate {
                        try { Process.Start("explorer.exe", LauncherCore.Quote(Path.GetDirectoryName(core.LogPath))); }
                        catch (Exception ex) { MessageBox.Show(ex.Message, "Could not open logs"); }
                    };
                    actions.Controls.Add(logs);
                }
                ControlBox = true;
                Show();
                Activate();
            }
        }

        [STAThread]
        private static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new StartupWindow());
        }
    }
}
