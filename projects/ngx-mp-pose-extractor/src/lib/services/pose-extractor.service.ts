import { EventEmitter, Injectable } from '@angular/core';
import {
  HAND_CONNECTIONS,
  Holistic,
  NormalizedLandmark,
  NormalizedLandmarkList,
  POSE_CONNECTIONS,
  POSE_LANDMARKS,
  POSE_LANDMARKS_LEFT,
  POSE_LANDMARKS_RIGHT,
  Results,
} from '@mediapipe/holistic';
import { drawConnectors, drawLandmarks, lerp } from '@mediapipe/drawing_utils';

/**
 * MediaPipe を用いて動画からポーズを抽出するためのサービス
 *
 * ※ シングルトンなサービスではないため、Component で providers に指定して使用することを想定
 */
@Injectable()
export class PoseExtractorService {
  public onResultsEventEmitter: EventEmitter<{
    mpResults: Results;
    sourceImageDataUrl: string;
    posePreviewImageDataUrl: string;
  }> = new EventEmitter();

  private holistic?: Holistic;

  private posePreviewCanvasElement?: HTMLCanvasElement;
  private posePreviewCanvasContext?: CanvasRenderingContext2D;

  private handPreviewCanvasElement?: HTMLCanvasElement;
  private handPreviewCanvasContext?: CanvasRenderingContext2D;

  private readonly IMAGE_JPEG_QUALITY = 0.8;

  constructor() {
    this.init();
  }

  public getPosePreviewMediaStream(): MediaStream | undefined {
    if (!this.posePreviewCanvasElement) return;
    return this.posePreviewCanvasElement.captureStream();
  }

  public getHandPreviewMediaStream(): MediaStream | undefined {
    if (!this.handPreviewCanvasElement) return;
    return this.handPreviewCanvasElement.captureStream();
  }

  public async onVideoFrame(videoElement: HTMLVideoElement) {
    if (!this.holistic) return;

    if (this.posePreviewCanvasElement) {
      this.posePreviewCanvasElement.width = videoElement.videoWidth;
      this.posePreviewCanvasElement.height = videoElement.videoHeight;
    }

    if (this.handPreviewCanvasElement) {
      this.handPreviewCanvasElement.width = videoElement.videoWidth;
      this.handPreviewCanvasElement.height = videoElement.videoHeight;
    }

    await this.holistic.send({ image: videoElement });
  }

  private init() {
    this.posePreviewCanvasElement = document.createElement('canvas');
    this.posePreviewCanvasContext =
      this.posePreviewCanvasElement.getContext('2d') || undefined;

    this.handPreviewCanvasElement = document.createElement('canvas');
    this.handPreviewCanvasContext =
      this.handPreviewCanvasElement.getContext('2d') || undefined;

    this.holistic = new Holistic({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
      },
    });

    this.holistic.setOptions({
      selfieMode: false,
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: true,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });

    this.holistic.onResults((results: Results) => {
      this.onResults(results);
    });
  }

  private onResults(results: Results) {
    if (
      !this.posePreviewCanvasElement ||
      !this.posePreviewCanvasContext ||
      !this.holistic
    )
      return;

    // 描画用に不必要なランドマークを除去
    let poseLandmarks: NormalizedLandmarkList = [];
    if (results.poseLandmarks) {
      poseLandmarks = JSON.parse(
        JSON.stringify(results.poseLandmarks)
      ) as NormalizedLandmarkList;
      this.removeElements(
        poseLandmarks,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22]
      );
    }

    // キャンバスを塗りつぶし
    this.posePreviewCanvasContext.save();
    this.posePreviewCanvasContext.clearRect(
      0,
      0,
      this.posePreviewCanvasElement.width,
      this.posePreviewCanvasElement.height
    );

    // 検出に使用したフレーム画像を描画
    this.posePreviewCanvasContext.drawImage(
      results.image,
      0,
      0,
      this.posePreviewCanvasElement.width,
      this.posePreviewCanvasElement.height
    );

    // 検出に使用したフレーム画像を保持
    const sourceImageDataUrl = this.posePreviewCanvasElement.toDataURL(
      'image/jpeg',
      this.IMAGE_JPEG_QUALITY
    );

    // 肘と手をつなぐ線を描画
    this.posePreviewCanvasContext.lineWidth = 5;
    if (poseLandmarks) {
      if (results.rightHandLandmarks) {
        this.posePreviewCanvasContext.strokeStyle = 'white';
        this.connect(this.posePreviewCanvasContext, [
          [
            poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW],
            results.rightHandLandmarks[0],
          ],
        ]);
      }
      if (results.leftHandLandmarks) {
        this.posePreviewCanvasContext.strokeStyle = 'white';
        this.connect(this.posePreviewCanvasContext, [
          [
            poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW],
            results.leftHandLandmarks[0],
          ],
        ]);
      }
    }

    // ポーズのプレビューを描画
    if (poseLandmarks) {
      drawConnectors(
        this.posePreviewCanvasContext,
        poseLandmarks,
        POSE_CONNECTIONS,
        { color: 'white' }
      );
      drawLandmarks(
        this.posePreviewCanvasContext,
        Object.values(POSE_LANDMARKS_LEFT).map((index) => poseLandmarks[index]),
        { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(255,138,0)' }
      );
      drawLandmarks(
        this.posePreviewCanvasContext,
        Object.values(POSE_LANDMARKS_RIGHT).map(
          (index) => poseLandmarks[index]
        ),
        { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(0,217,231)' }
      );
    }

    // 手のプレビューを描画
    drawConnectors(
      this.posePreviewCanvasContext,
      results.rightHandLandmarks,
      HAND_CONNECTIONS,
      { color: 'white' }
    );
    drawLandmarks(this.posePreviewCanvasContext, results.rightHandLandmarks, {
      color: 'white',
      fillColor: 'rgb(0,217,231)',
      lineWidth: 2,
      radius: (data: any) => {
        return lerp(data.from.z, -0.15, 0.1, 10, 1);
      },
    });
    drawConnectors(
      this.posePreviewCanvasContext,
      results.leftHandLandmarks,
      HAND_CONNECTIONS,
      { color: 'white' }
    );
    drawLandmarks(this.posePreviewCanvasContext, results.leftHandLandmarks, {
      color: 'white',
      fillColor: 'rgb(255,138,0)',
      lineWidth: 2,
      radius: (data: any) => {
        return lerp(data.from.z, -0.15, 0.1, 10, 1);
      },
    });

    // 手の領域のみのプレビューを生成
    if (this.handPreviewCanvasContext && this.handPreviewCanvasElement) {
      const HAND_PREVIEW_ZOOM = 3;
      const handPreviewBaseY = this.handPreviewCanvasElement.height / 2;

      this.handPreviewCanvasContext.clearRect(
        0,
        0,
        this.handPreviewCanvasElement.width,
        this.handPreviewCanvasElement.height
      );

      if (results.rightHandLandmarks) {
        const rect = this.getRectByLandmarks(
          results.rightHandLandmarks,
          results.image.width,
          results.image.height
        );

        let handPreviewX = 0;
        let handPreviewY = handPreviewBaseY - (rect[3] * HAND_PREVIEW_ZOOM) / 2;

        this.handPreviewCanvasContext.drawImage(
          this.posePreviewCanvasElement,
          rect[0] - 10,
          rect[1] - 10,
          rect[2] + 10,
          rect[3] + 10,
          handPreviewX,
          handPreviewY,
          rect[2] * HAND_PREVIEW_ZOOM,
          rect[3] * HAND_PREVIEW_ZOOM
        );
      }

      if (results.leftHandLandmarks) {
        const rect = this.getRectByLandmarks(
          results.leftHandLandmarks,
          results.image.width,
          results.image.height
        );

        let handPreviewX =
          this.handPreviewCanvasElement.width - rect[2] * HAND_PREVIEW_ZOOM;
        let handPreviewY = handPreviewBaseY - (rect[3] * HAND_PREVIEW_ZOOM) / 2;

        this.handPreviewCanvasContext.drawImage(
          this.posePreviewCanvasElement,
          rect[0] - 10,
          rect[1] - 10,
          rect[2] + 10,
          rect[3] + 10,
          handPreviewX,
          handPreviewY,
          rect[2] * HAND_PREVIEW_ZOOM,
          rect[3] * HAND_PREVIEW_ZOOM
        );
      }
    }

    // イベントを送出
    this.onResultsEventEmitter.emit({
      mpResults: results,
      sourceImageDataUrl: sourceImageDataUrl,
      posePreviewImageDataUrl: this.posePreviewCanvasElement.toDataURL(
        'image/jpeg',
        this.IMAGE_JPEG_QUALITY
      ),
    });

    // 完了
    this.posePreviewCanvasContext.restore();
  }

  private connect(
    ctx: CanvasRenderingContext2D,
    connectors: Array<[NormalizedLandmark, NormalizedLandmark]>
  ) {
    const canvas = ctx.canvas;
    for (const connector of connectors) {
      const from = connector[0];
      const to = connector[1];
      if (from && to) {
        if (
          from.visibility &&
          to.visibility &&
          (from.visibility < 0.1 || to.visibility < 0.1)
        ) {
          continue;
        }
        ctx.beginPath();
        ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
        ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
        ctx.stroke();
      }
    }
  }

  private removeElements(
    landmarks: NormalizedLandmarkList,
    elements: number[]
  ) {
    for (const element of elements) {
      delete landmarks[element];
    }
  }

  private getRectByLandmarks(landmarks: any[], width: number, height: number) {
    const leftHandLandmarksX = landmarks.map((landmark) => landmark.x * width);
    const leftHandLandmarksY = landmarks.map((landmark) => landmark.y * height);
    const minX = Math.min(...leftHandLandmarksX);
    const maxX = Math.max(...leftHandLandmarksX);
    const minY = Math.min(...leftHandLandmarksY);
    const maxY = Math.max(...leftHandLandmarksY);
    return [minX, minY, maxX - minX, maxY - minY];
  }
}
