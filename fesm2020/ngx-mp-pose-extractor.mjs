import * as i0 from '@angular/core';
import { Injectable, Component, NgModule, EventEmitter } from '@angular/core';
import { Holistic, POSE_LANDMARKS, POSE_CONNECTIONS, POSE_LANDMARKS_LEFT, POSE_LANDMARKS_RIGHT, HAND_CONNECTIONS } from '@mediapipe/holistic';
import { drawConnectors, drawLandmarks, lerp } from '@mediapipe/drawing_utils';

class NgxMpPoseExtractorService {
    constructor() { }
}
NgxMpPoseExtractorService.ɵfac = i0.ɵɵngDeclareFactory({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorService, deps: [], target: i0.ɵɵFactoryTarget.Injectable });
NgxMpPoseExtractorService.ɵprov = i0.ɵɵngDeclareInjectable({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorService, providedIn: 'root' });
i0.ɵɵngDeclareClassMetadata({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorService, decorators: [{
            type: Injectable,
            args: [{
                    providedIn: 'root'
                }]
        }], ctorParameters: function () { return []; } });

class NgxMpPoseExtractorComponent {
}
NgxMpPoseExtractorComponent.ɵfac = i0.ɵɵngDeclareFactory({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorComponent, deps: [], target: i0.ɵɵFactoryTarget.Component });
NgxMpPoseExtractorComponent.ɵcmp = i0.ɵɵngDeclareComponent({ minVersion: "14.0.0", version: "15.0.4", type: NgxMpPoseExtractorComponent, selector: "lib-ngx-mp-pose-extractor", ngImport: i0, template: `
    <p>
      ngx-mp-pose-extractor works!
    </p>
  `, isInline: true });
i0.ɵɵngDeclareClassMetadata({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorComponent, decorators: [{
            type: Component,
            args: [{ selector: 'lib-ngx-mp-pose-extractor', template: `
    <p>
      ngx-mp-pose-extractor works!
    </p>
  ` }]
        }] });

class NgxMpPoseExtractorModule {
}
NgxMpPoseExtractorModule.ɵfac = i0.ɵɵngDeclareFactory({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorModule, deps: [], target: i0.ɵɵFactoryTarget.NgModule });
NgxMpPoseExtractorModule.ɵmod = i0.ɵɵngDeclareNgModule({ minVersion: "14.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorModule, declarations: [NgxMpPoseExtractorComponent], exports: [NgxMpPoseExtractorComponent] });
NgxMpPoseExtractorModule.ɵinj = i0.ɵɵngDeclareInjector({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorModule });
i0.ɵɵngDeclareClassMetadata({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorModule, decorators: [{
            type: NgModule,
            args: [{
                    declarations: [NgxMpPoseExtractorComponent],
                    imports: [],
                    exports: [NgxMpPoseExtractorComponent],
                }]
        }] });

/**
 * MediaPipe を用いて動画からポーズを抽出するためのサービス
 *
 * ※ シングルトンなサービスではないため、Component で providers に指定して使用することを想定
 */
class PoseExtractorService {
    constructor() {
        this.onResultsEventEmitter = new EventEmitter();
        this.init();
    }
    getPosePreviewMediaStream() {
        if (!this.posePreviewCanvasElement)
            return;
        return this.posePreviewCanvasElement.captureStream();
    }
    getHandPreviewMediaStream() {
        if (!this.handPreviewCanvasElement)
            return;
        return this.handPreviewCanvasElement.captureStream();
    }
    async onVideoFrame(videoElement) {
        if (!this.holistic)
            return;
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
    init() {
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
        this.holistic.onResults((results) => {
            this.onResults(results);
        });
    }
    onResults(results) {
        if (!this.posePreviewCanvasElement ||
            !this.posePreviewCanvasContext ||
            !this.holistic)
            return;
        // 描画用に不必要なランドマークを除去
        let poseLandmarks = [];
        if (results.poseLandmarks) {
            poseLandmarks = JSON.parse(JSON.stringify(results.poseLandmarks));
            this.removeElements(poseLandmarks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22]);
        }
        // キャンバスを塗りつぶし
        this.posePreviewCanvasContext.save();
        this.posePreviewCanvasContext.clearRect(0, 0, this.posePreviewCanvasElement.width, this.posePreviewCanvasElement.height);
        // 検出に使用したフレーム画像を描画
        this.posePreviewCanvasContext.drawImage(results.image, 0, 0, this.posePreviewCanvasElement.width, this.posePreviewCanvasElement.height);
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
            drawConnectors(this.posePreviewCanvasContext, poseLandmarks, POSE_CONNECTIONS, { color: 'white' });
            drawLandmarks(this.posePreviewCanvasContext, Object.values(POSE_LANDMARKS_LEFT).map((index) => poseLandmarks[index]), { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(255,138,0)' });
            drawLandmarks(this.posePreviewCanvasContext, Object.values(POSE_LANDMARKS_RIGHT).map((index) => poseLandmarks[index]), { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(0,217,231)' });
        }
        // 手のプレビューを描画
        drawConnectors(this.posePreviewCanvasContext, results.rightHandLandmarks, HAND_CONNECTIONS, { color: 'white' });
        drawLandmarks(this.posePreviewCanvasContext, results.rightHandLandmarks, {
            color: 'white',
            fillColor: 'rgb(0,217,231)',
            lineWidth: 2,
            radius: (data) => {
                return lerp(data.from.z, -0.15, 0.1, 10, 1);
            },
        });
        drawConnectors(this.posePreviewCanvasContext, results.leftHandLandmarks, HAND_CONNECTIONS, { color: 'white' });
        drawLandmarks(this.posePreviewCanvasContext, results.leftHandLandmarks, {
            color: 'white',
            fillColor: 'rgb(255,138,0)',
            lineWidth: 2,
            radius: (data) => {
                return lerp(data.from.z, -0.15, 0.1, 10, 1);
            },
        });
        // 手の領域のみのプレビューを生成
        if (this.handPreviewCanvasContext && this.handPreviewCanvasElement) {
            const HAND_PREVIEW_ZOOM = 3;
            const handPreviewBaseY = this.handPreviewCanvasElement.height / 2;
            this.handPreviewCanvasContext.clearRect(0, 0, this.handPreviewCanvasElement.width, this.handPreviewCanvasElement.height);
            if (results.rightHandLandmarks) {
                const rect = this.getRectByLandmarks(results.rightHandLandmarks, results.image.width, results.image.height);
                let handPreviewX = 0;
                let handPreviewY = handPreviewBaseY - (rect[3] * HAND_PREVIEW_ZOOM) / 2;
                this.handPreviewCanvasContext.drawImage(this.posePreviewCanvasElement, rect[0] - 10, rect[1] - 10, rect[2] + 10, rect[3] + 10, handPreviewX, handPreviewY, rect[2] * HAND_PREVIEW_ZOOM, rect[3] * HAND_PREVIEW_ZOOM);
            }
            if (results.leftHandLandmarks) {
                const rect = this.getRectByLandmarks(results.leftHandLandmarks, results.image.width, results.image.height);
                let handPreviewX = this.handPreviewCanvasElement.width - rect[2] * HAND_PREVIEW_ZOOM;
                let handPreviewY = handPreviewBaseY - (rect[3] * HAND_PREVIEW_ZOOM) / 2;
                this.handPreviewCanvasContext.drawImage(this.posePreviewCanvasElement, rect[0] - 10, rect[1] - 10, rect[2] + 10, rect[3] + 10, handPreviewX, handPreviewY, rect[2] * HAND_PREVIEW_ZOOM, rect[3] * HAND_PREVIEW_ZOOM);
            }
        }
        // イベントを送出
        this.onResultsEventEmitter.emit({
            mpResults: results,
            posePreviewImageDataUrl: this.posePreviewCanvasElement.toDataURL('image/jpeg'),
        });
        // 完了
        this.posePreviewCanvasContext.restore();
    }
    connect(ctx, connectors) {
        const canvas = ctx.canvas;
        for (const connector of connectors) {
            const from = connector[0];
            const to = connector[1];
            if (from && to) {
                if (from.visibility &&
                    to.visibility &&
                    (from.visibility < 0.1 || to.visibility < 0.1)) {
                    continue;
                }
                ctx.beginPath();
                ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
                ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
                ctx.stroke();
            }
        }
    }
    removeElements(landmarks, elements) {
        for (const element of elements) {
            delete landmarks[element];
        }
    }
    getRectByLandmarks(landmarks, width, height) {
        const leftHandLandmarksX = landmarks.map((landmark) => landmark.x * width);
        const leftHandLandmarksY = landmarks.map((landmark) => landmark.y * height);
        const minX = Math.min(...leftHandLandmarksX);
        const maxX = Math.max(...leftHandLandmarksX);
        const minY = Math.min(...leftHandLandmarksY);
        const maxY = Math.max(...leftHandLandmarksY);
        return [minX, minY, maxX - minX, maxY - minY];
    }
}
PoseExtractorService.ɵfac = i0.ɵɵngDeclareFactory({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseExtractorService, deps: [], target: i0.ɵɵFactoryTarget.Injectable });
PoseExtractorService.ɵprov = i0.ɵɵngDeclareInjectable({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseExtractorService });
i0.ɵɵngDeclareClassMetadata({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseExtractorService, decorators: [{
            type: Injectable
        }], ctorParameters: function () { return []; } });

/*
 * Public API Surface of ngx-mp-pose-extractor
 */

/**
 * Generated bundle index. Do not edit.
 */

export { NgxMpPoseExtractorComponent, NgxMpPoseExtractorModule, NgxMpPoseExtractorService, PoseExtractorService };
//# sourceMappingURL=ngx-mp-pose-extractor.mjs.map
