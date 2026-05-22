"""
图像识别系统 GUI 界面：基于 tkinter 和 windnd，
点击图片框选择图片、拖拽图片、自动识别并展示 Top-5 结果。
"""

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import threading
import os

import windnd

from recognizer import ImageRecognizer

# ---------- 常量 ----------

WIN_TITLE = "图像识别系统"
WIN_WIDTH, WIN_HEIGHT = 1300, 650
IMG_DISPLAY_SIZE = (500, 400)
RESULT_TOP_K = 5

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff")
SUPPORTED_EXTS_STR = "、".join(ext.upper().lstrip(".") for ext in SUPPORTED_EXTS)

IMG_EXTENSIONS = (
    (f"图片文件 ({' '.join('*'+e for e in SUPPORTED_EXTS)})",
     " ".join("*"+e for e in SUPPORTED_EXTS)),
    ("所有文件", "*.*"),
)

BG_COLOR = "#f5f6fa"
HEADER_BG = "#2c3e50"
HEADER_FG = "#ffffff"
CARD_BG = "#ffffff"
ACCENT = "#3498db"
ACCENT_HOVER = "#2980b9"
TEXT_PRIMARY = "#2c3e50"
TEXT_SECONDARY = "#7f8c8d"

class MainWindow:
    """图像识别系统主界面。"""

    def __init__(self):
        self.recognizer = ImageRecognizer()
        self._current_image = None
        self._current_tk_image = None
        self._image_path = None
        self._results = []

        self.root = tk.Tk()
        self.root.title(WIN_TITLE)
        self.root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
        self.root.minsize(800, 550)
        self.root.configure(bg=BG_COLOR)

        self._build_ui()
        self._init_recognizer()

    # ---------- UI 构建 ----------

    def _build_ui(self):
        self._build_header()

        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        self._build_image_panel(main_frame)
        self._build_result_panel(main_frame)

        self.status_var = tk.StringVar(value="就绪 — 正在加载模型…")
        tk.Label(
            self.root, textvariable=self.status_var, anchor=tk.W,
            bg="#ecf0f1", fg=TEXT_SECONDARY, font=("微软雅黑", 9),
            padx=10, pady=4,
        ).pack(fill=tk.X, side=tk.BOTTOM)

    def _build_header(self):
        header = tk.Frame(self.root, bg=HEADER_BG, height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text="图像识别系统",
            bg=HEADER_BG, fg=HEADER_FG,
            font=("微软雅黑", 16, "bold"),
        ).pack(side=tk.LEFT, padx=20, pady=8)

    def _build_image_panel(self, parent):
        left_frame = tk.Frame(parent, bg=BG_COLOR)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        panel = tk.LabelFrame(
            left_frame, text=" 图片预览（点击选择 / 拖拽上传）",
            font=("微软雅黑", 11, "bold"),
            fg=TEXT_PRIMARY, bg=CARD_BG, padx=8, pady=8,
        )
        panel.pack(fill=tk.BOTH, expand=True)

        # 画布区域
        self.canvas_frame = tk.Frame(panel, bg="#ecf0f1", bd=1, relief=tk.SOLID)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.img_canvas = tk.Canvas(
            self.canvas_frame, bg="#ecf0f1", bd=0, highlightthickness=0,
            width=IMG_DISPLAY_SIZE[0], height=IMG_DISPLAY_SIZE[1],
            cursor="hand2",
        )
        self.img_canvas.pack(fill=tk.BOTH, expand=True)
        # 点击 Canvas 选择图片
        self.img_canvas.bind("<Button-1>", self._on_canvas_click)

        # 拖拽提示浮层（点击可选择图片）
        self.drop_frame = tk.Frame(
            self.img_canvas, bg="#d5e8f5", bd=2, relief=tk.GROOVE,
            cursor="hand2",
        )
        self.drop_frame.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)
        # 点击蓝色浮层选择图片
        self.drop_frame.bind("<Button-1>", self._on_canvas_click)
        for child in self.drop_frame.winfo_children():
            child.bind("<Button-1>", self._on_canvas_click)

        tk.Label(
            self.drop_frame, text="点击选择图片\n或\n将图片拖拽到此处",
            bg="#d5e8f5", fg=TEXT_SECONDARY,
            font=("微软雅黑", 14, "bold"), justify=tk.CENTER,
            cursor="hand2",
        ).pack(expand=True)

        tk.Label(
            self.drop_frame,
            text=f"支持格式：{SUPPORTED_EXTS_STR}",
            bg="#d5e8f5", fg=TEXT_SECONDARY,
            font=("微软雅黑", 9), cursor="hand2",
        ).pack(pady=(0, 10))

        # 底部信息栏：文件名 + 操作按钮
        bottom_frame = tk.Frame(panel, bg=CARD_BG, height=35)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0))

        self.hint_var = tk.StringVar(value="点击上方区域选择图片  或  拖拽图片到上方区域")
        tk.Label(
            bottom_frame, textvariable=self.hint_var,
            bg=CARD_BG, fg=TEXT_SECONDARY, font=("微软雅黑", 9),
        ).pack(side=tk.LEFT)

        tk.Label(
            bottom_frame, text=f"支持格式：{SUPPORTED_EXTS_STR}  ",
            bg=CARD_BG, fg=TEXT_SECONDARY, font=("微软雅黑", 8),
        ).pack(side=tk.LEFT)

        # 操作按钮放在图片框底部右侧
        self.recognize_btn = tk.Button(
            bottom_frame, text="开始识别", command=self._on_start_recognition,
            bg=ACCENT, fg="#fff", font=("微软雅黑", 9),
            bd=0, padx=10, pady=2, cursor="hand2",
            activebackground=ACCENT_HOVER, activeforeground="#fff",
        )
        self.recognize_btn.pack(side=tk.RIGHT, padx=2)

        tk.Button(
            bottom_frame, text="清除", command=self._on_clear,
            bg="#95a5a6", fg="#fff", font=("微软雅黑", 9),
            bd=0, padx=10, pady=2, cursor="hand2",
            activebackground="#7f8c8d", activeforeground="#fff",
        ).pack(side=tk.RIGHT, padx=2)

        # windnd 拖拽注册
        windnd.hook_dropfiles(self.img_canvas, func=self._on_drop)
        windnd.hook_dropfiles(self.canvas_frame, func=self._on_drop)
        windnd.hook_dropfiles(self.drop_frame, func=self._on_drop)
        windnd.hook_dropfiles(self.root, func=self._on_drop)

    def _bind_drop_frame_click(self):
        """递归绑定点击事件到 drop_frame 及其所有子控件。"""
        def bind_all(widget):
            widget.bind("<Button-1>", self._on_canvas_click)
            for child in widget.winfo_children():
                bind_all(child)
        bind_all(self.drop_frame)

    def _build_result_panel(self, parent):
        right_frame = tk.Frame(parent, bg=BG_COLOR, width=680)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(8, 0))
        right_frame.pack_propagate(False)

        panel = tk.LabelFrame(
            right_frame, text=f" 识别结果（Top-{RESULT_TOP_K}）",
            font=("微软雅黑", 11, "bold"),
            fg=TEXT_PRIMARY, bg=CARD_BG, padx=6, pady=6,
        )
        panel.pack(fill=tk.BOTH, expand=True)

        # 简单 Frame 承载卡片，无需 Canvas / 滚动条
        self.result_inner = tk.Frame(panel, bg=CARD_BG)
        self.result_inner.pack(fill=tk.BOTH, expand=True)

        self._show_result_placeholder()

    def _show_result_placeholder(self):
        for w in self.result_inner.winfo_children():
            w.destroy()
        tk.Label(
            self.result_inner, text="识别结果将在此显示",
            bg=CARD_BG, fg=TEXT_SECONDARY, font=("微软雅黑", 11),
            pady=40,
        ).pack()

    # ---------- 初始化识别引擎 ----------

    def _init_recognizer(self):
        def _load():
            try:
                self.status_var.set("正在加载模型…")
                self.recognizer.load_model()
                self.status_var.set("正在加载标签…")
                self.recognizer.load_labels()
                self.status_var.set("就绪 — 点击图片区域选择图片或直接拖入")
            except Exception as e:
                self.status_var.set(f"初始化失败：{e}")
                self.root.after(0, lambda: messagebox.showerror(
                    "初始化错误", f"模型或标签加载失败：\n{e}"))
        threading.Thread(target=_load, daemon=True).start()

    # ---------- 拖拽事件 ----------

    def _is_supported_image(self, filepath):
        return os.path.splitext(filepath)[1].lower() in SUPPORTED_EXTS

    def _on_drop(self, files):
        if not files:
            return
        filepath = files[0]
        if isinstance(filepath, bytes):
            filepath = filepath.decode("utf-8", errors="replace")

        if not os.path.isfile(filepath):
            return

        if not self._is_supported_image(filepath):
            messagebox.showwarning(
                "格式不支持",
                f"不支持的图片格式。\n\n"
                f"支持的文件类型：{SUPPORTED_EXTS_STR}\n"
                f"当前文件：{os.path.splitext(filepath)[1]}"
            )
            return

        self._load_image(filepath)
        self._start_recognition()

    # ---------- 点击选择图片 ----------

    def _on_canvas_click(self, event=None):
        """点击图片区域打开文件选择对话框。"""
        filepath = filedialog.askopenfilename(
            title="选择图片", filetypes=IMG_EXTENSIONS,
        )
        if not filepath:
            return
        self._load_image(filepath)

    # ---------- 按钮回调 ----------

    def _on_start_recognition(self):
        if not self._image_path:
            messagebox.showinfo("提示", "请先选择或拖入一张图片。")
            return
        self._start_recognition()

    def _on_clear(self):
        self._image_path = None
        self._current_image = None
        self._current_tk_image = None
        self._results = []
        self.img_canvas.delete("all")
        self.drop_frame.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)
        self.hint_var.set("点击上方区域选择图片  或  拖拽图片到上方区域")
        self._show_result_placeholder()
        self.status_var.set("就绪 — 点击图片区域选择图片或直接拖入")

    # ---------- 图片加载与显示 ----------

    def _load_image(self, filepath):
        try:
            img = Image.open(filepath)
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception as e:
            messagebox.showerror("加载失败", f"无法打开图片：\n{e}")
            return

        self._image_path = filepath
        self._current_image = img

        self.hint_var.set(f"当前：{os.path.basename(filepath)}")
        self._display_image(img)
        self.status_var.set("已加载 — 点击「开始识别」或拖入新图片自动识别")

    def _display_image(self, img):
        self.drop_frame.place_forget()

        cw = self.img_canvas.winfo_width()
        ch = self.img_canvas.winfo_height()
        if cw < 10:
            cw = IMG_DISPLAY_SIZE[0]
        if ch < 10:
            ch = IMG_DISPLAY_SIZE[1]

        img_w, img_h = img.size
        scale = min(cw / img_w, ch / img_h, 1.0)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        display = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self._current_tk_image = ImageTk.PhotoImage(display)

        self.img_canvas.delete("all")
        self.img_canvas.create_image(
            cw // 2, ch // 2, anchor=tk.CENTER, image=self._current_tk_image,
        )

    # ---------- 识别执行 ----------

    def _start_recognition(self):
        if not self._image_path:
            return

        if hasattr(self, "_recognizing") and self._recognizing:
            return
        self._recognizing = True

        filepath = self._image_path
        self.status_var.set("正在识别…")

        def _run():
            try:
                results, display_img = self.recognizer.predict(
                    filepath, top_k=RESULT_TOP_K)
                self.root.after(0, self._show_results, results, display_img)
            except Exception as e:
                self.root.after(0, self._on_recognition_error, str(e))
            finally:
                self._recognizing = False

        threading.Thread(target=_run, daemon=True).start()

    RANK_COLORS = {
        1: ("#f39c12", "#e67e22"),  # 金
        2: ("#95a5a6", "#7f8c8d"),  # 银
        3: ("#cd7f32", "#a0522d"),  # 铜
        4: ("#5dade2", "#3498db"),  # 蓝
        5: ("#5dade2", "#3498db"),
    }
    BAR_COLORS = {1: "#f1c40f", 2: "#bdc3c7", 3: "#e67e22",
                  4: "#3498db", 5: "#3498db"}

    def _create_result_card(self, parent, rank, zh_name, en_name, confidence):
        rank_bg, rank_fg = self.RANK_COLORS.get(rank, ("#95a5a6", "#7f8c8d"))
        bar_color = self.BAR_COLORS.get(rank, "#3498db")

        card = tk.Frame(parent, bg="white", bd=0, highlightthickness=0)
        card.pack(fill=tk.X, pady=3, padx=2)

        # 排名徽章
        badge = tk.Label(
            card, text=str(rank), bg=rank_bg, fg="white",
            font=("Arial", 18, "bold"), width=3, height=1,
        )
        badge.pack(side=tk.LEFT, padx=(4, 10), pady=6)

        # 中文名 + 英文名
        text_frame = tk.Frame(card, bg="white")
        text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)

        tk.Label(
            text_frame, text=zh_name,
            bg="white", fg=TEXT_PRIMARY, font=("微软雅黑", 12, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X)

        tk.Label(
            text_frame, text=en_name,
            bg="white", fg=TEXT_SECONDARY, font=("Consolas", 9),
            anchor=tk.W,
        ).pack(fill=tk.X)

        # 置信度区域
        conf_frame = tk.Frame(card, bg="white")
        conf_frame.pack(side=tk.RIGHT, padx=(0, 8), pady=4)

        # 进度条容器
        bar_container = tk.Frame(conf_frame, bg="#ecf0f1", width=160, height=14)
        bar_container.pack(side=tk.TOP, pady=(0, 2))
        bar_container.pack_propagate(False)

        bar_inner = tk.Frame(bar_container, bg=bar_color, width=0, height=14)
        bar_inner.place(x=0, y=0, relheight=1, width=0)
        bar_container.after(100, lambda: bar_inner.place_configure(
            width=int(160 * confidence / 100)))

        tk.Label(
            conf_frame, text=f"{confidence:.1f}%",
            bg="white", fg=rank_bg, font=("Consolas", 13, "bold"),
        ).pack(side=tk.TOP)

        return card

    def _show_results(self, results, display_img):
        self._results = results
        if display_img:
            self._display_image(display_img)

        # 清空旧卡片
        for w in self.result_inner.winfo_children():
            w.destroy()

        # 创建 Top-5 卡片
        for rank, zh_name, en_name, confidence in results:
            self._create_result_card(
                self.result_inner, rank, zh_name, en_name, confidence)

        top1 = results[0]
        self.status_var.set(f"识别完成 — Top-1: {top1[1]}（{top1[3]}%）")

    def _on_recognition_error(self, msg):
        self.status_var.set(f"识别失败：{msg}")
        messagebox.showerror("识别错误", f"识别过程发生错误：\n{msg}")

    # ---------- 启动 ----------

    def run(self):
        self.root.mainloop()
