import argparse
import threading
import queue
import cv2
import os
import time

from engine.engine_factory import EngineFactory, EngineType
from video_source.video_source_factory import VideoSourceFactory, SourceType

def serial_io_worker(tx_queue: queue.Queue):
    """Isolated background thread handling asynchronous serial communications."""
    while True:
        try:
            data = tx_queue.get(timeout=1.0)
            if data == "SHUTDOWN":
                break
        except queue.Empty:
            continue

def network_io_worker(network_queue: queue.Queue):
    """Isolated background thread handling network socket operations."""
    while True:
        try:
            data = network_queue.get(timeout=1.0)
            if data == "SHUTDOWN":
                break
        except queue.Empty:
            continue

def video_writer_worker(write_queue: queue.Queue, output_path: str, fps: float, width: int, height: int, total_frames: int):
    """Dedicated worker that processes the frame queue sequentially *after* AI pipeline finishes."""
    print(f"\n[Saver] Starting video generation phase. Target path: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    saved_count = 0
    try:
        while True:
            frame = write_queue.get()
            if isinstance(frame, str) and frame == "SHUTDOWN":
                break
            
            video_writer.write(frame)
            saved_count += 1
            progress_pct = (saved_count / total_frames) * 100 if total_frames > 0 else 0
            print(f" -> Saving Frame Queue: {saved_count}/{total_frames} ({progress_pct:.1f}%) complete...", end="\r", flush=True)
    finally:
        video_writer.release()
        print(f"\n[Saver] Video file completely sealed and finalized successfully.")

def main():
    parser = argparse.ArgumentParser(description="Clean sequential phase multi-threaded tracking orchestration pipeline.")
    parser.add_argument('--engine', type=str, choices=[e.value for e in EngineType], default='gpu',
                        help="Inference engine hardware backend choice. (Default: gpu)")
    parser.add_argument('--source', type=str, choices=[s.value for s in SourceType], default='file',
                        help="Video data ingestion source choice. (Default: file)")
    parser.add_argument('--input', type=str, default='media/sample.mp4',
                        help="Target file path used if source is 'file'. (Default: media/sample.mp4)")
    parser.add_argument('--model', type=str, default=None,
                        help="Optional custom model weights override path. Omit to use class defaults.")
    parser.add_argument('--cam_idx', type=int, default=0,
                        help="System hardware index mapping used if source is 'camera'.")
    parser.add_argument('--debug', action='store_true', default=False,
                        help="Save tracked output to media/output_tracked.mp4 instead of displaying a window.")
    args = parser.parse_args()

    selected_engine_type = EngineType(args.engine)
    selected_source_type = SourceType(args.source)

    engine = EngineFactory.create_engine(selected_engine_type, model_path=args.model)
    video_source = VideoSourceFactory.create_source(selected_source_type, file_path=args.input, camera_idx=args.cam_idx)

    engine.initialize()
    video_source.start()

    serial_tx_queue = queue.Queue()
    network_tx_queue = queue.Queue()
    video_write_queue = queue.Queue()

    serial_thread = threading.Thread(target=serial_io_worker, args=(serial_tx_queue,), daemon=True)
    network_thread = threading.Thread(target=network_io_worker, args=(network_tx_queue,), daemon=True)
    serial_thread.start()
    network_thread.start()

    total_frames, fps, width, height = 0, 30.0, 1920, 1080
    if args.source == 'file':
        probe = cv2.VideoCapture(args.input)
        total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        probe.release()

    print("\n[Pipeline] System fully initialized. Frame processing active. Press Ctrl+C to exit.\n")

    try:
        frame_count = 0
        processing_start_time = time.time()

        while True:
            frame_read_start = time.time()
            success, frame = video_source.read_frame()
            if not success:
                break

            frame_count += 1
            boxes, track_ids, labels = engine.track_frame(frame)

            if track_ids:
                serial_tx_queue.put({"target_count": len(track_ids), "ids": track_ids})
                network_tx_queue.put({"status": "PROCESSING", "tracked": track_ids})

                for box, track_id, label in zip(boxes, track_ids, labels):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if args.debug:
                video_write_queue.put(frame.copy())
                elapsed_time = time.time() - frame_read_start
                current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
                progress_pct = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"[AI Processing] Frame: {frame_count}/{total_frames} ({progress_pct:.1f}%) | Performance: {current_fps:.2f} FPS", end="\r", flush=True)
            else:
                cv2.imshow("Multi-Threaded Edge Compute Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        total_processing_time = time.time() - processing_start_time
        average_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        print(f"\n\n[Pipeline] Video processing finished. Average Tracking Speed: {average_fps:.2f} FPS")

    except KeyboardInterrupt:
        print("\n[Pipeline] Execution interrupted by user.")
    finally:
        print("[Pipeline] Tearing down input hardware feeds...")
        video_source.release()
        cv2.destroyAllWindows()
        
        if args.debug and frame_count > 0:
            output_path = os.path.join("media", "output_tracked.mp4")
            writer_thread = threading.Thread(
                target=video_writer_worker, 
                args=(video_write_queue, output_path, fps, width, height, frame_count),
                daemon=True
            )
            writer_thread.start()
            video_write_queue.put("SHUTDOWN")
            writer_thread.join()

        serial_tx_queue.put("SHUTDOWN")
        network_tx_queue.put("SHUTDOWN")
        serial_thread.join(timeout=1.0)
        network_thread.join(timeout=1.0)
        print("[Pipeline] System safe shutdown complete.")

if __name__ == '__main__':
    main()
