#!/usr/bin/env python3
"""
测试 VLM 检测集成

此代码是用于代替 perception_ros_wrapper 方案中原有的 YOLO 检测部分，目的三达到 VLM 去识别出“叉”进行bbox生成

Usage:
    # 1. 测试 VLM 检测器（独立运行）
    python test_vlm_detection.py --image path/to/image.png --query "Where is the fork?"
    
    # 2. 测试 ROS2 服务调用
    python test_vlm_detection.py --ros2-service
    
    # 3. 测试完整流程（YOLO -> VLM）
    python test_vlm_detection.py --full-pipeline
"""
import sys
import argparse
import json
import time
import os
from pathlib import Path

# 添加路径
SCRIPT_DIR = Path(__file__).parent
# This repo's helper scripts live in ./scripts
MS_SWIFT_DIR = SCRIPT_DIR / "scripts"

sys.path.insert(0, str(MS_SWIFT_DIR))


# ---------------------------
# Prompting (domain anchoring)
# ---------------------------
DOMAIN_DEFINITION = (
    "In this task, the target 'tomato fork' refers to the tomato axillary bud/sucker at the leaf axil "
    "(the junction between the main stem and the leaf petiole). It is a pruning target, not a utensil fork."
)

# Keep the output format aligned with our fine-tuned training (Qwen3-VL grounding tokens).
OUTPUT_FORMAT = (
    "Return only one tight bounding box in the format: "
    "<|box_start|>(x1,y1),(x2,y2)<|box_end|> using norm1000 coordinates."
)


def build_domain_query(user_prompt: str, *, add_domain_definition: bool = True, bbox_only: bool = True) -> str:
    """Compose a domain-anchored query for small VLMs."""
    parts = [user_prompt.rstrip().rstrip(".") + "."]
    if add_domain_definition:
        parts.append(DOMAIN_DEFINITION)
    if bbox_only:
        parts.append(OUTPUT_FORMAT)
    return " ".join(parts)


def _load_camera_rgb_topic(camera_name: str) -> str:
    """Load RGB topic for a given camera from configs/camera.yaml."""
    import yaml

    cfg_path = SCRIPT_DIR / "configs" / "camera.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"camera.yaml not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if camera_name not in cfg:
        raise KeyError(f"camera '{camera_name}' not found in {cfg_path}")
    topic = cfg[camera_name].get("rgb_topic")
    if not topic:
        raise ValueError(f"rgb_topic missing for camera '{camera_name}' in {cfg_path}")
    return topic


def _wait_for_one_rgb_frame(node, topic: str, timeout_sec: float = 3.0):
    """Subscribe and wait for one RGB frame; supports Image and CompressedImage."""
    import time as _time
    import rclpy
    from sensor_msgs.msg import Image as ROSImage
    from sensor_msgs.msg import CompressedImage

    msg_holder = {"msg": None, "type": None}

    def cb_img(msg):
        msg_holder["msg"] = msg
        msg_holder["type"] = "raw"

    def cb_cmp(msg):
        msg_holder["msg"] = msg
        msg_holder["type"] = "compressed"

    # Try raw Image first (most common).
    sub = node.create_subscription(ROSImage, topic, cb_img, 10)
    start = _time.time()
    while rclpy.ok() and msg_holder["msg"] is None and (_time.time() - start) < timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.1)

    if msg_holder["msg"] is None:
        node.destroy_subscription(sub)
        # Fallback to compressed.
        sub = node.create_subscription(CompressedImage, topic, cb_cmp, 10)
        start = _time.time()
        while rclpy.ok() and msg_holder["msg"] is None and (_time.time() - start) < timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_subscription(sub)
    return msg_holder["msg"], msg_holder["type"]


def _ros_img_to_bgr(msg, msg_type: str):
    """Convert ROS Image/CompressedImage to BGR numpy array."""
    import numpy as np
    import cv2

    # Prefer cv_bridge if available.
    try:
        from cv_bridge import CvBridge

        bridge = CvBridge()
        if msg_type == "raw":
            # Use msg.encoding; commonly bgr8/rgb8.
            # cv_bridge will convert to bgr8 if we ask it.
            return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if msg_type == "compressed":
            array = np.frombuffer(msg.data, dtype=np.uint8)
            return cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception:
        pass

    # Fallback conversion without cv_bridge (limited support).
    if msg_type != "raw":
        array = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    h, w = msg.height, msg.width
    enc = (msg.encoding or "").lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("bgr8", "rgb8"):
        img = data.reshape((h, w, 3))
        if enc == "rgb8":
            img = img[:, :, ::-1]
        return img
    if enc in ("mono8",):
        img = data.reshape((h, w))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported encoding without cv_bridge: {msg.encoding}")


def _draw_norm1000_bbox(bgr, bbox_norm1000, color=(0, 0, 255), thickness=3):
    """Draw norm1000 bbox on BGR image (in-place) and return pixel bbox."""
    import cv2

    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_norm1000]
    x1_px = int(round(x1 / 1000.0 * w))
    y1_px = int(round(y1 / 1000.0 * h))
    x2_px = int(round(x2 / 1000.0 * w))
    y2_px = int(round(y2 / 1000.0 * h))

    # Clamp.
    x1_px = max(0, min(w - 1, x1_px))
    x2_px = max(0, min(w - 1, x2_px))
    y1_px = max(0, min(h - 1, y1_px))
    y2_px = max(0, min(h - 1, y2_px))

    cv2.rectangle(bgr, (x1_px, y1_px), (x2_px, y2_px), color, thickness)
    return (x1_px, y1_px, x2_px, y2_px)


def test_vlm_detector():
    """测试 VLM 检测器"""
    print("=" * 60)
    print("测试 VLM 检测器")
    print("=" * 60)
    
    from vlm_detector_wrapper import detect, detect_and_draw
    import tempfile
    
    # 测试图像
    test_image = SCRIPT_DIR / "Screenshot-4.png"
    if not test_image.exists():
        print(f"[Error] 测试图像不存在: {test_image}")
        return False
    
    # 测试查询
    # These prompts are intentionally domain-anchored to disambiguate "fork" for small models.
    queries = [
        build_domain_query("Locate the branching junction on the tomato stem."),
        build_domain_query("Locate the axillary bud (sucker) at the leaf axil for pruning."),
        build_domain_query("Identify the target zone for robotic pruning (de-suckering)."),
        build_domain_query("Find the tomato fork, ignoring leaves and background."),
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        print("-" * 40)
        
        result = detect(
            image_path=str(test_image),
            query=query
        )
        
        if result.get("success"):
            bbox = result["bbox"]
            print(f"✅ 检测成功!")
            print(f"   bbox (norm1000): {bbox}")
            try:
                import cv2

                img = cv2.imread(str(test_image))
                h, w = img.shape[:2]
                x1_px = round(bbox[0] / 1000 * w)
                y1_px = round(bbox[1] / 1000 * h)
                x2_px = round(bbox[2] / 1000 * w)
                y2_px = round(bbox[3] / 1000 * h)
                print(f"   bbox (px): [{x1_px}, {y1_px}, {x2_px}, {y2_px}]  image={w}x{h}")
            except Exception:
                pass
        else:
            print(f"❌ 检测失败: {result.get('error', 'Unknown error')}")
            print(f"   原始输出: {result.get('raw_output', '')[:200]}")
    
    return True


def test_ros2_service():
    """测试 ROS2 服务调用"""
    print("\n" + "=" * 60)
    print("测试 ROS2 服务调用")
    print("=" * 60)
    
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    
    # 尝试导入 robot_interface（需要先编译）
    try:
        from robot_interface.srv import VLMDetection
    except ImportError:
        print("[Warn] robot_interface 未编译，请先编译 ROS2 包")
        print("  cd /path/to/robot_ws && colcon build --packages-select robot_interface")
        return False
    
    class TestClient(Node):
        def __init__(self):
            super().__init__('vlm_test_client')
            self.cli = self.create_client(VLMDetection, '/vlm_detection')
            while not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('/vlm_detection 服务不可用，等待中...')
            self.req = VLMDetection.Request()
    
    try:
        rclpy.init()
        client = TestClient()
        
        # 发送请求
        client.req.camera = "left_hand_eye"
        client.req.query = build_domain_query("Locate the axillary bud (sucker) at the leaf axil.")
        
        print("发送 VLM 检测请求...")
        future = client.cli.call_async(client.req)
        rclpy.spin_until_future_complete(client, future)
        
        response = future.result()
        if response:
            print(f"响应: success={response.success}, message={response.message}")
            if response.success:
                print(f"  bbox: {list(response.bbox)}")
                # Visualize on the current RGB frame.
                try:
                    cam = client.req.camera
                    topic = _load_camera_rgb_topic(cam)
                    msg, msg_type = _wait_for_one_rgb_frame(client, topic, timeout_sec=3.0)
                    if msg is None:
                        print(f"[Warn] 未获取到 {cam} RGB 帧用于可视化（topic={topic}）")
                    else:
                        import cv2
                        from datetime import datetime

                        bgr = _ros_img_to_bgr(msg, msg_type)
                        bbox_px = _draw_norm1000_bbox(bgr, response.bbox)
                        vis_dir = str(SCRIPT_DIR / "ms-swift" / "vis")
                        os.makedirs(vis_dir, exist_ok=True)
                        out_path = os.path.join(
                            vis_dir,
                            f"vlm_detection_{cam}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png",
                        )
                        cv2.imwrite(out_path, bgr)
                        print(f"  bbox_px: {bbox_px}")
                        print(f"  vis: {out_path}")
                except Exception as e:
                    print(f"[Warn] 可视化失败: {e}")
            else:
                print(f"  raw_output: {response.raw_output[:200]}")
        
        client.destroy_node()
        rclpy.shutdown()
        return response.success if response else False
        
    except Exception as e:
        print(f"[Error] ROS2 服务调用失败: {e}")
        return False


def test_full_pipeline():
    """测试完整流程：YOLO -> VLM"""
    print("\n" + "=" * 60)
    print("测试完整流程 (YOLO -> VLM)")
    print("=" * 60)
    
    print("""
完整流程说明:
1. YOLO 检测器检测目标，获取 bbox
2. VLM 在 bbox 区域内进行细粒度检测
3. 输出更精确的 bbox

当前状态:
- YOLO 检测器: 需要启用 AuroraDetection (已注释)
- VLM 检测器: 已就绪

要启用完整流程，需要:
1. 取消 cv_robot.py 中 AuroraDetection 的注释
2. 编译 robot_interface 包
3. 启动 run_node.py --mode hybrid
""")


def main():
    parser = argparse.ArgumentParser(description="测试 VLM 检测集成")
    parser.add_argument("--image", type=str, help="测试图像路径")
    parser.add_argument(
        "--query",
        type=str,
        default="Locate the axillary bud (sucker) at the leaf axil (stem-petiole junction).",
        help="查询问题（会自动追加领域定义与bbox-only格式约束）",
    )
    parser.add_argument(
        "--no_domain_definition",
        action="store_true",
        help="不追加领域定义（不推荐，小模型容易把fork理解成餐叉/支架交叉）",
    )
    parser.add_argument(
        "--free_form",
        action="store_true",
        help="不追加bbox-only格式约束（不推荐，会产生解释性废话）",
    )
    parser.add_argument("--ros2-service", action="store_true",
                        help="测试 ROS2 服务调用")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="测试完整流程")
    parser.add_argument("--all", action="store_true",
                        help="运行所有测试")
    
    args = parser.parse_args()
    
    # 检查是否有测试图像
    test_image = args.image
    if not test_image:
        test_image = SCRIPT_DIR / "Screenshot-4.png"
    test_image = Path(test_image)
    
    success = True
    
    if args.all or args.ros2_service:
        success = test_ros2_service() and success
    
    if args.all or args.full_pipeline:
        test_full_pipeline()
    
    if args.all or (not args.ros2_service and not args.full_pipeline):
        if test_image.exists():
            # 修改 vlm_detector.py 的默认图像路径进行测试
            import tempfile
            import shutil
            import cv2
            
            # 创建临时测试图像
            temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            shutil.copy(test_image, temp_image.name)
            
            print(f"测试图像: {temp_image.name}")
            
            # 直接运行 VLM 检测器脚本
            import subprocess
            venv_python = SCRIPT_DIR / "venv_msswift" / "bin" / "python"
            final_query = build_domain_query(
                args.query,
                add_domain_definition=not args.no_domain_definition,
                bbox_only=not args.free_form,
            )
            result = subprocess.run(
                [
                    str(venv_python),
                    str(MS_SWIFT_DIR / "vlm_detector.py"),
                    "--image", temp_image.name,
                    "--query", final_query,
                    "--json"
                ],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            print(f"Return code: {result.returncode}")
            print(f"Stdout: {result.stdout[:500] if result.stdout else 'Empty'}")
            print(f"Stderr: {result.stderr[:500] if result.stderr else 'Empty'}")
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    output = json.loads(result.stdout)
                    print("\n" + "=" * 60)
                    print("VLM 检测结果")
                    print("=" * 60)
                    print(json.dumps(output, indent=2))
                    
                    # 正确计算像素坐标（从归一化 0-1000 转换）
                    if output.get("success") and output.get("bbox"):
                        import cv2
                        img = cv2.imread(temp_image.name)
                        h, w = img.shape[:2]
                        bbox = output["bbox"]
                        # bbox 是归一化坐标 (0-1000)，转换为像素坐标
                        x1_px = round(bbox[0] / 1000 * w)
                        y1_px = round(bbox[1] / 1000 * h)
                        x2_px = round(bbox[2] / 1000 * w)
                        y2_px = round(bbox[3] / 1000 * h)
                        print(f"\n像素坐标: x1={x1_px}, y1={y1_px}, x2={x2_px}, y2={y2_px}")
                        print(f"图像尺寸: {w}x{h}")
                        # 保存可视化
                        vis_dir = SCRIPT_DIR / "ms-swift" / "vis"
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        vis_img = img.copy()
                        cv2.rectangle(vis_img, (x1_px, y1_px), (x2_px, y2_px), (0, 0, 255), 3)
                        vis_path = vis_dir / f"vlm_detection_{Path(test_image).stem}.png"
                        cv2.imwrite(str(vis_path), vis_img)
                        print(f"可视化已保存: {vis_path}")
                except json.JSONDecodeError as e:
                    print(f"\n[Warn] JSON 解析失败，尝试处理原始输出")
                    # 如果不是 JSON，尝试提取 bbox
                    output_text = result.stdout
                    print(f"原始输出: {output_text}")
            else:
                print(f"[Error] 脚本执行失败")
                success = False
            
            # 清理
            try:
                os.unlink(temp_image.name)
            except:
                pass
        else:
            print(f"[Warn] 测试图像不存在: {test_image}")
            print("请使用 --image 指定测试图像")
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 测试完成")
    else:
        print("❌ 测试失败")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
