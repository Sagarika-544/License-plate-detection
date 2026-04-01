#!/usr/bin/env python3
# =============================================================================
# main.py
# Smart License Plate Detection System
# Entry point — starts the webcam/video stream and runs the pipeline live.
# =============================================================================

import cv2
import argparse
import sys

from src.pipeline.pipeline    import Pipeline
from src.pipeline.video_stream import VideoStream
from src.utils.logger         import log_system, log_error


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart License Plate Detection System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source:\n  0, 1, 2 ... = webcam index\n  path/to/video.mp4 = video file"
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to YOLOv8 weights file (default: auto-detect)"
    )
    parser.add_argument(
        "--save", default=None,
        help="Save annotated output to this video file (e.g. output.mp4)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headless (no OpenCV window) — useful for servers"
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # Resolve source — integer for webcam, string for file
    source = int(args.source) if args.source.isdigit() else args.source

    log_system(f"Starting | Source: {source}")

    # ---- Initialise pipeline ----
    try:
        pipeline = Pipeline(model_path=args.model)
    except Exception as e:
        log_error("Failed to initialise pipeline", e)
        sys.exit(1)

    # ---- Set up optional video writer ----
    writer = None

    # ---- Start stream ----
    stream = VideoStream(source=source)

    print("\n" + "=" * 55)
    print("  Smart License Plate Detection System")
    print("  Press  Q  to quit")
    print("=" * 55 + "\n")

    try:
        stream.open()

        # Set up video writer after opening the stream (need frame dimensions)
        if args.save:
            fps    = stream.fps or 30
            width  = stream.frame_width
            height = stream.frame_height
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
            log_system(f"Saving output to: {args.save}")

        for frame, frame_number in stream.stream():

            # Run full pipeline on this frame
            annotated = pipeline.process(frame, frame_number)

            # Save frame to output video (if requested)
            if writer:
                writer.write(annotated)

            # Display in OpenCV window (unless headless mode)
            if not args.no_display:
                cv2.imshow("Smart License Plate Detection", annotated)

                # Q or ESC to quit
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    log_system("Quit key pressed.")
                    break

    except KeyboardInterrupt:
        log_system("Interrupted by user (Ctrl+C).")

    except Exception as e:
        log_error("Unexpected error in main loop", e)

    finally:
        stream.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        log_system("System shut down cleanly.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()