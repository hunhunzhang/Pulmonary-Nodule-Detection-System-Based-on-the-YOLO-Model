import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pydicom
import imageio
import os
import sys
import cv2
from ultralytics import YOLO
import threading
from tkinter import ttk
import traceback
import multiprocessing

#调整窗宽窗位
def apply_window(image, window_center, window_width):
    img = image.astype(np.float32)
    img = (img - window_center + 0.5 * window_width) / window_width
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

class LungNoduleDetectionApp:
    def __init__(self, root):
        # 设置环境变量防止多进程问题
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['YOLO_VERBOSE'] = 'False'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['TORCH_NUM_THREADS'] = '1'
        os.environ['PYTORCH_DISABLE_PER_OP_PROFILING'] = '1'
        os.environ['ULTRALYTICS_SETTINGS'] = os.path.join(os.path.expanduser('~'), '.ultralytics', 'settings.yaml')
        
        self.root = root
        self.root.title("肺结节检测系统")
        self.root.geometry("1200x800")
        self.root.bind("<Configure>", self.on_resize)

        # 初始化图像显示相关属性
        self.image_displayed = None  
        self.nodule_displayed = None 
        self.dicom_pixel_array = None

        # 获取代码所在目录（支持打包后的可执行文件）
        if getattr(sys, 'frozen', False):
            # 如果是打包后的可执行文件
            self.base_dir = os.path.dirname(sys.executable)
        else:
            # 如果是源码运行
            self.base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

        # 工具栏
        self.toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED, bg="white")
        self.toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")  # 工具栏占据两列，宽度与窗口一致

        # 设置工具栏各列权重，使其随窗口宽度自适应
        for i in range(8):  
            self.toolbar.grid_columnconfigure(i, weight=1)

        # 工具栏中的组件使用 grid 布局，并设置sticky="ew"
        self.select_model_btn = tk.Button(self.toolbar, text="选择模型", command=self.select_model_btn, bg="white", relief="ridge")
        self.select_model_btn.grid(row=0, column=0, padx=5, pady=2, sticky="ew")

        self.upload_button = tk.Button(self.toolbar, text="选择图像", command=self.upload_image, width=20, bg="white", relief="ridge")
        self.upload_button.grid(row=0, column=2, padx=5, pady=2, sticky="ew")

        self.detect_button = tk.Button(self.toolbar, text="检测结节", command=self.detect_nodules, width=20, bg="white", relief="ridge")
        self.detect_button.grid(row=0, column=3, padx=5, pady=2, sticky="ew")
        # 批量处理按钮
        self.batch_button = tk.Button(self.toolbar, text="批量处理", command=self.batch_detect, bg="white", relief="ridge")
        self.batch_button.grid(row=0, column=7, padx=5, pady=2, sticky="ew")

        # 窗宽滑动条
        self.window_width_frame = tk.Frame(self.toolbar, bg="white")
        self.window_width_frame.grid(row=0, column=4, padx=5, pady=2, sticky="ew")

        self.window_width_label = tk.Label(self.window_width_frame, text="窗宽:", bg="white")
        self.window_width_label.pack(side=tk.LEFT, padx=5, pady=2)

        self.window_width_spinbox = tk.Spinbox(self.window_width_frame, from_=100, to=2000, width=5, justify=tk.CENTER,
                                               command=self.update_window_width_from_spinbox)
        self.window_width_spinbox.delete(0, tk.END)
        self.window_width_spinbox.insert(0, "1500")  # 默认值
        self.window_width_spinbox.bind("<Return>", self.update_window_width_from_spinbox_event)
        self.window_width_spinbox.pack(side=tk.LEFT, padx=5, pady=2)

        self.window_width_slider = tk.Scale(self.window_width_frame, from_=0, to=2000, orient=tk.HORIZONTAL, length=150,
                                            command=self.update_window_width_from_slider, showvalue=False)
        self.window_width_slider.set(1500)
        self.window_width_slider.pack(side=tk.LEFT, padx=5, pady=5)

        # 窗位滑动条
        self.window_level_frame = tk.Frame(self.toolbar, bg="white")
        self.window_level_frame.grid(row=0, column=5, padx=5, pady=2, sticky="ew")

        self.window_level_label = tk.Label(self.window_level_frame, text="窗位:", bg="white")
        self.window_level_label.pack(side=tk.LEFT, padx=5, pady=2)

        self.window_level_spinbox = tk.Spinbox(self.window_level_frame, from_=0, to=2000, width=5, justify=tk.CENTER,
                                               command=self.update_window_level_from_spinbox)
        self.window_level_spinbox.delete(0, tk.END)
        self.window_level_spinbox.insert(0, "-600")  # 默认值
        self.window_level_spinbox.bind("<Return>", self.update_window_level_from_spinbox_event)
        self.window_level_spinbox.pack(side=tk.LEFT, padx=5, pady=2)

        self.window_level_slider = tk.Scale(self.window_level_frame, from_=-1000, to=1000, orient=tk.HORIZONTAL, length=150,
                                            command=self.update_window_level_from_slider, showvalue=False)
        self.window_level_slider.set(-600)
        self.window_level_slider.pack(side=tk.LEFT, padx=5, pady=5)

        # 置信度阈值设置（仅保留Spinbox）
        self.conf_frame = tk.Frame(self.toolbar, bg="white")
        self.conf_frame.grid(row=0, column=6, padx=5, pady=2)

        self.conf_label = tk.Label(self.conf_frame, text="置信度阈值:", bg="white")
        self.conf_label.pack(side=tk.LEFT, padx=5, pady=2)

        self.conf_spinbox = tk.Spinbox(
            self.conf_frame, from_=0.01, to=1.0, increment=0.01, width=5, justify=tk.CENTER,
            command=self.update_conf_from_spinbox
        )
        self.conf_spinbox.delete(0, tk.END)
        self.conf_spinbox.insert(0, "0.25")  # 默认值
        self.conf_spinbox.bind("<Return>", self.update_conf_from_spinbox_event)
        self.conf_spinbox.pack(side=tk.LEFT, padx=5, pady=2)

        self.conf_threshold = 0.25  # 当前置信度阈值

        # 主体布局
        # 文件资源管理器容器：单独位于左侧
        self.file_explorer_frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN,bg="white")
        self.file_explorer_frame.grid(row=1, column=0, rowspan=3, padx=2, pady=2, sticky="nsew")

        # 文件路径标签
        self.file_path_label = tk.Label(self.file_explorer_frame, text="资源管理器 ", anchor="center", bg="white", relief="flat")
        self.file_path_label.pack(fill=tk.X, padx=1, pady=1)

        # 当前显示图像名称标签（采用两个Label拼接）
        frame = tk.Frame(self.file_explorer_frame, bg="white")
        frame.pack(fill=tk.X, padx=1, pady=1)

        self.current_image_prefix = tk.Label(frame, text="当前图像：", anchor="w", bg="white", fg="black",relief="flat")
        self.current_image_prefix.pack(side=tk.LEFT)

        self.current_image_name = tk.Label(frame, text="无", anchor="w", bg="white", fg="blue", relief="flat")
        self.current_image_name.pack(side=tk.LEFT)

        # 文件列表框
        self.file_listbox = tk.Listbox(self.file_explorer_frame, selectmode=tk.SINGLE,exportselection=0)
        self.file_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        # 右侧容器：包含图像显示区域、预测结果区域和信息面板
        self.right_frame = tk.Frame(self.root)
        self.right_frame.grid(row=1, column=1, rowspan=3, padx=2, pady=2, sticky="nsew")

        # 图像显示区域的 Frame
        self.image_frame = tk.Frame(self.right_frame, relief=tk.SUNKEN, bd=1)
        self.image_frame.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")

        self.image_panel = tk.Label(self.image_frame, text="选择图像", bg="black", fg="white")
        self.image_panel.bind("<MouseWheel>", lambda event: self.on_mouse_wheel(event, self.image_panel))
        self.image_panel.bind("<ButtonPress-1>", self.on_mouse_drag_start)
        self.image_panel.bind("<B1-Motion>", lambda event: self.on_mouse_drag(event, self.image_panel))
        self.image_panel.bind("<ButtonRelease-1>", self.on_mouse_drag_end)
        self.image_panel.bind("<Button-3>", lambda event: self.on_mouse_right_click(event, self.image_panel))
        self.image_panel.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 预测结果区域的 Frame
        self.nodule_frame = tk.Frame(self.right_frame, relief=tk.SUNKEN, bd=1)
        self.nodule_frame.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

        self.nodule_panel = tk.Label(self.nodule_frame, text="预测结果", bg="black", fg="white")
        self.nodule_panel.bind("<MouseWheel>", lambda event: self.on_mouse_wheel(event, self.nodule_panel))
        self.nodule_panel.bind("<ButtonPress-1>", self.on_mouse_drag_start)
        self.nodule_panel.bind("<B1-Motion>", lambda event: self.on_mouse_drag(event, self.nodule_panel))
        self.nodule_panel.bind("<ButtonRelease-1>", self.on_mouse_drag_end)
        self.nodule_panel.bind("<Button-3>", lambda event: self.on_mouse_right_click(event, self.nodule_panel))
        self.nodule_panel.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 信息面板
        self.info_frame = tk.Frame(self.right_frame, relief=tk.SUNKEN, bd=1, height=150)  # 固定高度为 150
        self.info_frame.grid(row=1, column=0, columnspan=2, padx=2, pady=2, sticky="nsew")

        # 信息面板工具栏
        self.info_toolbar = tk.Frame(self.info_frame, bd=1, relief="flat", bg="white")
        self.info_toolbar.pack(fill=tk.X, padx=2, pady=2)

        self.total_metrics_button = tk.Button(self.info_toolbar, text="总信息", command=self.show_total_metrics, relief="ridge")
        self.total_metrics_button.pack(side=tk.LEFT, padx=0, pady=0)

        self.detailed_info_button = tk.Button(self.info_toolbar, text="详细信息", command=self.show_detailed_info, relief="ridge")
        self.detailed_info_button.pack(side=tk.LEFT, padx=0, pady=0)

        self.output_button = tk.Button(self.info_toolbar, text="输出", command=self.show_output_panel, relief="ridge")
        self.output_button.pack(side=tk.LEFT, padx=0, pady=0)

        # 信息内容区域
        self.info_content_frame = tk.Frame(self.info_frame, height=100)  # 固定高度为 100
        self.info_content_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=0)

        # 总指标界面（改为 Listbox）
        self.total_metrics_panel = tk.Listbox(self.info_content_frame, selectmode=tk.SINGLE, width=80, height=8, relief="flat", borderwidth=1)
        self.total_metrics_panel.pack(fill=tk.BOTH, expand=True)
        self.total_metrics_panel.pack_forget()  # 初始隐藏

        # 详细信息界面（改为 Listbox）
        self.detailed_info_panel = tk.Listbox(self.info_content_frame, selectmode=tk.SINGLE, width=80, height=8, relief="flat", borderwidth=1)
        self.detailed_info_panel.pack(fill=tk.BOTH, expand=True)
        self.detailed_info_panel.pack_forget()  # 初始隐藏

        # 输出界面
        self.output_panel = tk.Text(self.info_content_frame, wrap=tk.WORD, width=80, height=11, state=tk.DISABLED, relief="flat", borderwidth=1)
        self.output_panel.pack(fill=tk.BOTH, expand=True)
        self.output_panel.pack_forget()  # 初始隐藏

        # 默认显示总指标界面
        self.show_total_metrics()

        # 调整布局权重
        self.root.grid_rowconfigure(0, weight=0)  # 工具栏占据较少空间
        self.root.grid_rowconfigure(1, weight=1)  # 文件资源管理器和右侧内容占据较多空间
        self.root.grid_columnconfigure(0, weight=0)  # 文件资源管理器占较少空间
        self.root.grid_columnconfigure(1, weight=3)  # 右侧内容占较多空间

        self.right_frame.grid_rowconfigure(0, weight=1)  # 图像区域
        self.right_frame.grid_rowconfigure(1, weight=0)  #信息面板区域
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)
        
        self.yolo_model_path = os.path.join(self.base_dir, "yolo1.pt")  # 默认 YOLO 模型路径（使用绝对路径）
        self.yolo_model = None  # 缓存YOLO模型实例


        # 工具栏有8列，需调整权重
        for i in range(8):  # 工具栏列数从原来的7调整为8
            self.toolbar.grid_columnconfigure(i, weight=1)

        # 图像放缩和拖拽相关属性
        self.image_scale = 1.0  # 初始缩放比例
        self.image_offset_x = 0  # 图像水平偏移
        self.image_offset_y = 0  # 图像垂直偏移
        self.drag_start_x = None  # 拖拽起始点X
        self.drag_start_y = None  # 拖拽起始点Y

        # 模型名称标签
        self.model_name_label = tk.Label(self.toolbar, text="yolo1.pt", fg="blue", anchor="w", bg="white")
        self.model_name_label.grid(row=0, column=1, padx=5, pady=2, sticky="w")


    def visualize_prediction(self, image_path, model_path, conf_threshold=0.25, use_cached_model=False):
        """
        使用 YOLO 模型预测结节，并保存预测结果图像和结节信息。
        """
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                self.append_output(f"错误: 模型文件不存在 {model_path}")
                return
            
            # 设置环境变量以避免多进程问题
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            os.environ['YOLO_VERBOSE'] = 'False'
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['TORCH_NUM_THREADS'] = '1'
            os.environ['PYTORCH_DISABLE_PER_OP_PROFILING'] = '1'
            
            # 使用缓存的模型实例或重新加载
            if use_cached_model and self.yolo_model is not None and hasattr(self, 'cached_model_path') and self.cached_model_path == model_path:
                model = self.yolo_model
            else:
                # 加载模型时禁用多进程
                model = YOLO(model_path)
                if use_cached_model:
                    self.yolo_model = model
                    self.cached_model_path = model_path
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                self.append_output(f"错误: 无法读取图片 {image_path}")
                return
                
            # 转换为 RGB 格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 模型预测（禁用多进程和详细输出）
            results = model.predict(image, conf=conf_threshold, verbose=False, device='cpu')
            
            # 创建 /result 文件夹
            result_dir = os.path.join(os.path.dirname(image_path), "result")
            os.makedirs(result_dir, exist_ok=True)
            
            # 如果没有检测结果，保存原始图像并清空结节信息文件
            if not results[0].boxes:
                output_path = os.path.join(result_dir, os.path.basename(image_path).replace(".png", "_detect.png"))
                Image.fromarray(image).save(output_path)  # 使用 PIL 保存 RGB 图像
                self.append_output(f"未检测到结节，已保存原始图像到: {output_path}")
                # 清空结节信息文件
                info_path = os.path.join(result_dir, os.path.basename(image_path).replace(".png", "_info.txt"))
                with open(info_path, "w") as f:
                    f.write("")  # 写入空内容，清空文件
                return
                
            # 保存结节信息
            nodule_info = []
            vis_image = image.copy()
            nodule_idx = 1
            for result in results:
                for box in result.boxes:
                    # 获取框坐标
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    
                    # 获取类别和置信度
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    label = f"{result.names[cls_id]} {conf:.2f}"
                    
                    # 保存结节信息
                    nodule_info.append({
                        "coordinates": (x_min, y_min, x_max, y_max),
                        "class": result.names[cls_id],
                        "confidence": conf
                    })
                    
                    # 绘制边界框和标签
                    color = (255, 0, 0)  # 红色
                    vis_image = cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 1)
                    # 添加编号
                    vis_image = cv2.putText(
                        vis_image, f"{nodule_idx}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                    )
                    nodule_idx += 1
        
            # 保存结果图像
            output_path = os.path.join(result_dir, os.path.basename(image_path).replace(".png", "_detect.png"))
            Image.fromarray(vis_image).save(output_path)  # 使用 PIL 保存 RGB 图像
            self.append_output(f"检测结果已保存到: {output_path}")

            # 保存结节信息到文件
            info_path = os.path.join(result_dir, os.path.basename(image_path).replace(".png", "_info.txt"))
            with open(info_path, "w") as f:
                for info in nodule_info:
                    f.write(f"{info}\n")
            self.append_output(f"结节信息已保存到: {info_path}")
            
        except Exception as e:
            self.append_output(f"预测过程中发生错误: {str(e)}")
            import traceback
            self.append_output(f"详细错误信息: {traceback.format_exc()}")


    def select_model_btn(self):
        """选择 YOLO 模型权重文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("YOLO 模型文件", "*.pt")],
            initialdir=self.base_dir
        )
        if not file_path:
            return
        try:
            self.yolo_model_path = file_path
            # 清除缓存的模型实例，强制重新加载
            self.yolo_model = None
            if hasattr(self, 'cached_model_path'):
                delattr(self, 'cached_model_path')
            
            self.model_name_label.config(text=os.path.basename(file_path))  # 更新模型名称标签
            messagebox.showinfo("提示", f"模型已成功加载：{os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("DICOM Files", "*.dcm")],
            initialdir=self.base_dir
        )
        if not file_path:
            return
        try:
            dicom_data = pydicom.dcmread(file_path)
            pixel_array = dicom_data.pixel_array.astype(np.int16)
            if 'RescaleSlope' in dicom_data and 'RescaleIntercept' in dicom_data:
                pixel_array = pixel_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
            self.dicom_pixel_array = pixel_array  # 保存原始像素数据
            self.image_path = file_path  # 保存当前图像路径

            # 重置缩放因子和偏移量
            self.image_scale = 1.0
            self.image_offset_x = 0
            self.image_offset_y = 0

            # 初始窗宽窗位应用
            self.refresh_image_panel()
            self.refresh_nodule_panel_blank()
            self.root.after(100, self.force_panels_same_size)

            # 调用 populate_file_explorer 方法，填充文件资源管理器
            folder_path = os.path.dirname(file_path)
            self.populate_file_explorer(folder_path)

            # 更新当前图像名称标签（只更新蓝色部分）
            self.current_image_name.config(text=os.path.basename(file_path))

            # 高亮当前文件在列表中
            for idx in range(self.file_listbox.size()):
                if self.file_listbox.get(idx) == os.path.basename(file_path):
                    self.file_listbox.selection_clear(0, tk.END)
                    self.file_listbox.selection_set(idx)
                    self.file_listbox.activate(idx)
                    self.file_listbox.see(idx)
                    break

            # 检查是否存在预测信息
            result_dir = os.path.join(os.path.dirname(self.image_path), "result")
            result_image_path = os.path.join(result_dir, os.path.basename(self.image_path).replace(".dcm", "_detect.png"))
            info_path = os.path.join(result_dir, os.path.basename(self.image_path).replace(".dcm", "_info.txt"))

            if os.path.exists(result_image_path) and os.path.exists(info_path):
                # 加载预测信息
                self.load_nodule_info(info_path)
                # 加载预测图像
                nodule_image = Image.open(result_image_path)  # 加载 RGB 图像
                self.display_image(nodule_image, self.nodule_panel)  # 显示预测结果图像
            else:
                # 显示没有检测到结节的信息
                self.show_no_nodule_info(detected=False)  # 还没有进行任何检测

        except Exception as e:
            messagebox.showerror("错误", f"无法加载图像: {str(e)}")

    def on_window_change(self, event=None):
        if self.dicom_pixel_array is not None:
            self.refresh_image_panel()

    def refresh_image_panel(self):
        """刷新图像显示区域，应用窗宽窗位并保存为 PNG 图像"""
        window_center = self.window_level_slider.get()
        window_width = self.window_width_slider.get()
        png_image = apply_window(self.dicom_pixel_array, window_center, window_width)
        
        # 保存为临时 PNG 图像，名称与源图像名称相同
        temp_png_name = os.path.basename(self.image_path).replace(".dcm", ".png")
        self.temp_png_path = os.path.join(os.path.dirname(self.image_path), temp_png_name)
        imageio.imwrite(self.temp_png_path, png_image)
        
        # 检查 /result 文件夹是否存在预测结果图像
        result_dir = os.path.join(os.path.dirname(self.image_path), "result")
        os.makedirs(result_dir, exist_ok=True)
        result_image_path = os.path.join(result_dir, os.path.basename(self.temp_png_path).replace(".png", "_detect.png"))
        
        if os.path.exists(result_image_path):
            # 如果预测结果图像存在，直接加载并显示
            nodule_image = Image.open(result_image_path).convert('RGB')
            self.display_image(nodule_image, self.nodule_panel)  # 更新预测结果显示区域
            #print(f"预测结果已存在，直接加载: {result_image_path}")
        else:
            # 如果预测结果图像不存在，显示空白区域
            self.refresh_nodule_panel_blank()

        # 加载并显示原始图像
        self.image = Image.open(self.temp_png_path).convert('L')
        self.display_image(self.image, self.image_panel)

    def detect_nodules(self):
        """用户点击检测结节时，重新生成预测结果图像"""
        if self.image_path is None or not hasattr(self, 'temp_png_path'):
            messagebox.showerror("错误", "请先上传图像并刷新显示")
            return

        try:
            result_dir = os.path.join(os.path.dirname(self.image_path), "result")
            os.makedirs(result_dir, exist_ok=True)

            result_image_path = os.path.join(result_dir, os.path.basename(self.temp_png_path).replace(".png", "_detect.png"))
            # 读取当前置信度阈值
            conf = self.conf_threshold
            self.visualize_prediction(self.temp_png_path, self.yolo_model_path, conf_threshold=conf)
            if os.path.exists(result_image_path):
                nodule_image = Image.open(result_image_path)  # 加载 RGB 图像
                self.display_image(nodule_image, self.nodule_panel)  # 显示预测结果图像

                info_path = os.path.join(result_dir, os.path.basename(self.temp_png_path).replace(".png", "_info.txt"))
                if os.path.exists(info_path):
                    self.load_nodule_info(info_path)  # 刷新信息面板
                else:
                    self.show_no_nodule_info(detected=True)
            else:
                messagebox.showerror("错误", "检测失败，未生成结果图像")
            # 检测完成后弹窗提示
            messagebox.showinfo("提示", "检测完成！")

            # 切换到总指标界面
            self.show_total_metrics()

        except Exception as e:
            messagebox.showerror("错误", f"检测失败: {str(e)}")

    def load_nodule_info(self, info_path):
        self.detailed_info_panel.delete(0, tk.END)
        self.nodule_info = []
        try:
            with open(info_path, "r") as f:
                for idx, line in enumerate(f, 1):
                    info = eval(line.strip())
                    self.nodule_info.append(info)
                    coordinates = info["coordinates"]
                    cls = info["class"]
                    conf = info["confidence"]
                    size = (coordinates[2] - coordinates[0]) * (coordinates[3] - coordinates[1])
                    # 行首加编号
                    self.detailed_info_panel.insert(
                        tk.END,
                        f"序号: {idx}. 位置: {coordinates}, 类别: {cls}, 置信度: {conf:.3f}, 大小: {size:.2f}"
                    )
            self.total_metrics_panel.delete(0, tk.END)
            self.total_metrics_panel.insert(tk.END, f"结节个数: {len(self.nodule_info)}")
        except Exception as e:
            messagebox.showerror("错误", f"无法加载结节信息: {str(e)}")

    def on_nodule_select(self, event):
        pass  

    def refresh_nodule_panel_blank(self):
        """显示一张与左侧 panel 同样大小的灰色图"""
        width = self.image_panel.winfo_width()
        height = self.image_panel.winfo_height()
        if width < 10 or height < 10:
            width, height = 400, 400
        blank = Image.new('L', (width, height), color='black')
        self.display_image(blank, self.nodule_panel)
        
    def display_image(self, image, panel):
        """在指定的 panel 中显示图像，应用缩放和偏移"""
        width = panel.winfo_width()
        height = panel.winfo_height()
        if width < 10 or height < 10:
            width, height = 400, 400

        # 应用缩放
        scaled_width = int(image.width * self.image_scale)
        scaled_height = int(image.height * self.image_scale)
        resized_image = image.resize((scaled_width, scaled_height))

        # 应用偏移
        offset_x = max(0, min(self.image_offset_x, scaled_width - width))
        offset_y = max(0, min(self.image_offset_y, scaled_height - height))
        cropped_image = resized_image.crop((offset_x, offset_y, offset_x + width, offset_y + height))

        img = ImageTk.PhotoImage(cropped_image)
        panel.config(image=img)
        panel.image = img  # 确保引用保留，防止图像被垃圾回收
        if panel == self.image_panel:
            self.image_displayed = image
        elif panel == self.nodule_panel:
            self.nodule_displayed = image

    def on_resize(self, event):
        width = self.image_panel.winfo_width()
        height = self.image_panel.winfo_height()
        if self.image_displayed:
            img = self.image_displayed.resize((width, height))
            self.display_image(img, self.image_panel)
        if self.nodule_displayed:
            img = self.nodule_displayed.resize((width, height))
            self.display_image(img, self.nodule_panel)

    def force_panels_same_size(self):
        # 刷新两侧panel，保证大小一致
        if self.image_displayed:
            self.display_image(self.image_displayed, self.image_panel)
        if self.nodule_displayed:
            self.display_image(self.nodule_displayed, self.nodule_panel)

    def update_window_width_from_slider(self, value):
        # 更新窗宽值输入框
        self.window_width_spinbox.delete(0, tk.END)
        self.window_width_spinbox.insert(0, str(int(float(value))))
        self.on_window_change()

    def update_window_level_from_slider(self, value):
        # 更新窗位值输入框
        self.window_level_spinbox.delete(0, tk.END)
        self.window_level_spinbox.insert(0, str(int(float(value))))
        self.on_window_change()

    def update_conf_from_slider(self, value):
        self.conf_spinbox.delete(0, tk.END)
        self.conf_spinbox.insert(0, f"{float(value):.2f}")
        self.conf_threshold = float(value)

    def update_window_width_from_spinbox(self):
        """当窗宽 Spinbox 的值改变时，更新滑动条的位置"""
        try:
            value = int(self.window_width_spinbox.get())
            self.window_width_slider.set(value)
            self.on_window_change()
        except ValueError:
            pass

    def update_window_level_from_spinbox(self):
        """当窗位 Spinbox 的值改变时，更新滑动条的位置"""
        try:
            value = int(self.window_level_spinbox.get())
            self.window_level_slider.set(value)
            self.on_window_change()
        except ValueError:
            pass

    def update_conf_from_spinbox(self):
        try:
            value = float(self.conf_spinbox.get())
            self.conf_threshold = value
        except ValueError:
            pass


    def update_window_width_from_spinbox_event(self, event):
        """当窗宽 Spinbox 的值改变时，按下 Enter 键更新滑动条的位置并移除焦点"""
        try:
            value = int(self.window_width_spinbox.get())
            self.window_width_slider.set(value)
            self.on_window_change()
        except ValueError:
            pass
        finally:
            # 移除焦点
            self.root.focus()

    def update_window_level_from_spinbox_event(self, event):
        """当窗位 Spinbox 的值改变时，按下 Enter 键更新滑动条的位置并移除焦点"""
        try:
            value = int(self.window_level_spinbox.get())
            self.window_level_slider.set(value)
            self.on_window_change()
        except ValueError:
            pass
        finally:
            # 移除焦点
            self.root.focus()

    def update_conf_from_spinbox_event(self, event):
        try:
            value = float(self.conf_spinbox.get())
            self.conf_threshold = value
        except ValueError:
            pass
        finally:
            self.root.focus()
            
    def enable_entry_edit(self, event):
        # 双击输入框时启用编辑
        entry = event.widget
        entry.select_range(0, tk.END)

    def show_total_metrics(self):
        """显示总指标界面"""
        self.output_panel.pack_forget()  # 隐藏输出界面
        self.output_button.config(relief="raised")  # 设置输出按钮为凹陷状态
        self.detailed_info_panel.pack_forget()  # 隐藏详细信息界面
        self.detailed_info_button.config(relief="raised")  # 设置详细信息按钮为平坦状态
        self.total_metrics_button.config(relief="sunken")  # 设置总指标按钮为凹陷状态
        self.total_metrics_panel.pack(fill=tk.BOTH, expand=True)  # 显示总指标界面

    def show_detailed_info(self):
        """显示详细信息界面"""
        self.output_panel.pack_forget()  # 隐藏输出界面
        self.output_button.config(relief="raised")
        self.total_metrics_panel.pack_forget()  # 隐藏总指标界面
        self.total_metrics_button.config(relief="raised")  # 设置总指标按钮为平坦状态
        self.detailed_info_button.config(relief="sunken")  # 设置详细信息按钮为凹陷状态
        self.detailed_info_panel.pack(fill=tk.BOTH, expand=True)  # 显示详细信息界面

    def show_output_panel(self):
        """显示输出界面"""
        self.total_metrics_panel.pack_forget()  # 隐藏总指标界面
        self.total_metrics_button.config(relief="raised")
        self.detailed_info_panel.pack_forget()  # 隐藏详细信息界面
        self.detailed_info_button.config(relief="raised")
        self.output_button.config(relief="sunken")  # 设置输出按钮为凹陷状态
        self.output_panel.pack(fill=tk.BOTH, expand=True)  # 显示输出界面
        

    def populate_file_explorer(self, folder_path):
        """填充文件资源管理器，显示指定文件夹中的所有文件"""
        self.file_listbox.delete(0, tk.END)  # 清空列表框
        try:
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(".dcm"):  # 仅显示 DICOM 文件
                    self.file_listbox.insert(tk.END, file_name)
        except Exception as e:
            messagebox.showerror("错误", f"无法读取文件夹内容: {str(e)}")

    def on_file_select(self, event):
        selected_index = self.file_listbox.curselection()
        if not selected_index:
            return
        selected_file = self.file_listbox.get(selected_index)
        # 更新当前图像名称标签（只更新蓝色部分）
        self.current_image_name.config(text=selected_file)

        folder_path = os.path.dirname(self.image_path)  # 使用当前图像路径的文件夹
        selected_file_path = os.path.join(folder_path, selected_file)
        try:
            # 加载 DICOM 图像
            dicom_data = pydicom.dcmread(selected_file_path)
            pixel_array = dicom_data.pixel_array.astype(np.int16)
            if 'RescaleSlope' in dicom_data and 'RescaleIntercept' in dicom_data:
                pixel_array = pixel_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
            self.dicom_pixel_array = pixel_array  # 保存原始像素数据
            self.image_path = selected_file_path  # 更新当前图像路径

            # 重置缩放因子和偏移量
            self.image_scale = 1.0
            self.image_offset_x = 0
            self.image_offset_y = 0

            # 刷新图像显示
            self.refresh_image_panel()
            self.refresh_nodule_panel_blank()
            self.root.after(100, self.force_panels_same_size)

            # 检查是否存在预测信息
            result_dir = os.path.join(os.path.dirname(self.image_path), "result")
            result_image_path = os.path.join(result_dir, os.path.basename(self.image_path).replace(".dcm", "_detect.png"))
            info_path = os.path.join(result_dir, os.path.basename(self.image_path).replace(".dcm", "_info.txt"))

            if os.path.exists(result_image_path) and os.path.exists(info_path):
                # 加载预测信息
                self.load_nodule_info(info_path)
                # 加载预测图像
                nodule_image = Image.open(result_image_path)  # 加载 RGB 图像
                self.display_image(nodule_image, self.nodule_panel)  # 显示预测结果图像
            else:
                # 显示没有检测到结节的信息
                self.show_no_nodule_info(detected=False)  # 还没有进行任何检测
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图像: {str(e)}")

    def show_no_nodule_info(self, detected=False):
        """显示没有检测到结节的信息"""
        # 更新总指标界面
        self.total_metrics_panel.delete(0, tk.END)  # 清空 Listbox
        if detected:
            self.total_metrics_panel.insert(tk.END, "结节个数: 0")  # 检测了但没有检测到结节
        else:
            self.total_metrics_panel.insert(tk.END, "请先进行检测获取信息")  # 还没有进行任何检测

        # 更新详细信息界面
        self.detailed_info_panel.delete(0, tk.END)  # 清空 Listbox
        self.detailed_info_panel.insert(tk.END, "None")  # 插入 "None"

        # 切换到总指标界面
        self.show_total_metrics()

    def refresh_nodule_panel(self):
        """刷新预测结果图像"""
        result_dir = os.path.join(os.path.dirname(self.image_path), "result")
        result_image_path = os.path.join(result_dir, os.path.basename(self.image_path).replace(".dcm", "_detect.png"))

        if os.path.exists(result_image_path):
            # 加载预测结果图像
            nodule_image = Image.open(result_image_path)  # 加载 RGB 图像
            self.display_image(nodule_image, self.nodule_panel)  # 显示预测结果图像

    def batch_detect(self):
        """对当前文件夹下所有 DICOM 文件进行批量检测（带进度条和终止按钮）"""
        if not hasattr(self, "image_path") or not self.image_path:
            messagebox.showwarning("提示", "请先选择一张图像或上传图像以确定批量处理目录。")
            return
        folder_path = os.path.dirname(self.image_path)
        dcm_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".dcm")]
        if not dcm_files:
            messagebox.showinfo("提示", "当前文件夹下没有 DICOM 文件。")
            return

        if not messagebox.askyesno("批量处理", f"将对 {len(dcm_files)} 个 DICOM 文件进行批量检测，是否继续？"):
            return

        # 创建进度窗口
        self.batch_stop_flag = False
        progress_win = tk.Toplevel(self.root)
        progress_win.title("批量处理进度")
        progress_win.geometry("400x200")
        progress_label = tk.Label(progress_win, text="检测中，请稍候...")
        progress_label.pack(pady=5)
        progress_bar = ttk.Progressbar(progress_win, maximum=len(dcm_files), length=350)
        progress_bar.pack(pady=5)
        output_text = tk.Text(progress_win, height=6, width=48, state=tk.DISABLED)
        output_text.pack(pady=5)

        def append_output(msg):
            output_text.config(state=tk.NORMAL)
            output_text.insert(tk.END, msg + "\n")
            output_text.see(tk.END)
            output_text.config(state=tk.DISABLED)
            progress_win.update()

        def stop_batch():
            self.batch_stop_flag = True
            append_output("用户请求终止，正在停止...")

        stop_btn = tk.Button(progress_win, text="终止", command=stop_batch, fg="red")
        stop_btn.pack(pady=5)

        def batch_thread():
            completed = 0
            # 预加载模型（只加载一次）
            try:
                append_output("正在加载YOLO模型...")
                conf = self.conf_threshold
                # 创建一个临时图像来初始化模型
                temp_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
                temp_path = os.path.join(folder_path, "temp_init.png")
                imageio.imwrite(temp_path, temp_image)
                self.visualize_prediction(temp_path, self.yolo_model_path, conf_threshold=conf, use_cached_model=True)
                if os.path.exists(temp_path):
                    os.remove(temp_path)  # 删除临时文件
                append_output("模型加载完成，开始批量检测...")
            except Exception as e:
                append_output(f"模型加载失败: {e}")
                return
                
            for idx, file_name in enumerate(dcm_files, 1):
                if self.batch_stop_flag:
                    append_output("批量处理已终止。")
                    break
                try:
                    append_output(f"[{idx}/{len(dcm_files)}] 正在处理 {file_name}")
                    file_path = os.path.join(folder_path, file_name)
                    dicom_data = pydicom.dcmread(file_path)
                    pixel_array = dicom_data.pixel_array.astype(np.int16)
                    if 'RescaleSlope' in dicom_data and 'RescaleIntercept' in dicom_data:
                        pixel_array = pixel_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept

                    window_center = self.window_level_slider.get()
                    window_width = self.window_width_slider.get()
                    png_image = apply_window(pixel_array, window_center, window_width)
                    temp_png_name = os.path.basename(file_path).replace(".dcm", ".png")
                    temp_png_path = os.path.join(folder_path, temp_png_name)
                    imageio.imwrite(temp_png_path, png_image)

                    conf = self.conf_threshold
                    # 使用缓存的模型实例进行预测
                    self.visualize_prediction(temp_png_path, self.yolo_model_path, conf_threshold=conf, use_cached_model=True)
                    append_output(f"[{idx}/{len(dcm_files)}] {file_name} 检测完成")
                    completed += 1
                except Exception as e:
                    append_output(f"[{idx}/{len(dcm_files)}] {file_name} 检测失败: {e}")
                progress_bar['value'] = idx
                progress_win.update()
            if self.batch_stop_flag:
                append_output("批量处理已终止。")
            else:
                append_output(f"批量处理完成，已完成 {completed} 个 DICOM 文件的检测。")
            stop_btn.config(state=tk.DISABLED)
            progress_label.config(text="批量处理结束")
            progress_win.after(3000, progress_win.destroy)

        threading.Thread(target=batch_thread, daemon=True).start()

    def on_mouse_wheel(self, event, panel):
        """鼠标滚轮放缩图像，以当前显示区域中心为中心"""
        scale_factor = 1.1 if event.delta > 0 else 0.9
        new_scale = max(0.1, min(self.image_scale * scale_factor, 10))  # 限制缩放范围

        # 获取显示区域的宽高
        width = panel.winfo_width()
        height = panel.winfo_height()

        # 计算缩放中心点（显示区域中心）
        center_x = self.image_offset_x + width // 2
        center_y = self.image_offset_y + height // 2

        # 计算新的偏移量，确保缩放以中心点为基准
        scaled_width = int(self.image_displayed.width * new_scale)
        scaled_height = int(self.image_displayed.height * new_scale)
        self.image_offset_x = max(0, min(center_x * new_scale / self.image_scale - width // 2, scaled_width - width))
        self.image_offset_y = max(0, min(center_y * new_scale / self.image_scale - height // 2, scaled_height - height))

        # 更新缩放比例
        self.image_scale = new_scale

        # 刷新显示
        self.display_image(self.image_displayed if panel == self.image_panel else self.nodule_displayed, panel)

    def on_mouse_drag_start(self, event):
        """记录拖拽起始点"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_mouse_drag(self, event, panel):
        """拖拽图像"""
        if self.drag_start_x is None or self.drag_start_y is None:
            return

        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.drag_start_x = event.x
        self.drag_start_y = event.y

        # 更新偏移量
        self.image_offset_x -= dx
        self.image_offset_y -= dy

        # 限制偏移范围
        scaled_width = int(self.image_displayed.width * self.image_scale)
        scaled_height = int(self.image_displayed.height * self.image_scale)
        self.image_offset_x = max(0, min(self.image_offset_x, scaled_width - panel.winfo_width()))
        self.image_offset_y = max(0, min(self.image_offset_y, scaled_height - panel.winfo_height()))

        self.display_image(self.image_displayed if panel == self.image_panel else self.nodule_displayed, panel)

    def on_mouse_drag_end(self, event):
        """结束拖拽"""
        self.drag_start_x = None
        self.drag_start_y = None

    def on_mouse_right_click(self, event, panel):
        """右键点击，图像回归原始大小"""
        self.image_scale = 1.0  # 恢复缩放比例
        self.image_offset_x = 0  # 恢复水平偏移
        self.image_offset_y = 0  # 恢复垂直偏移
        self.display_image(self.image_displayed if panel == self.image_panel else self.nodule_displayed, panel)

    def append_output(self, message):
        """将内容追加到输出界面"""
        self.output_panel.config(state=tk.NORMAL)
        self.output_panel.insert(tk.END, message + "\n")
        self.output_panel.see(tk.END)  # 滚动到最后
        self.output_panel.config(state=tk.DISABLED)

if __name__ == "__main__":
    # 防止打包后程序启动多个实例的问题
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 强制设置多进程启动方法为spawn（Windows推荐）
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过，忽略错误
    
    # 设置环境变量来防止多进程问题
    os.environ['TORCH_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['PYTORCH_DISABLE_PER_OP_PROFILING'] = '1'
    
    root = tk.Tk()
    app = LungNoduleDetectionApp(root)
    root.mainloop()