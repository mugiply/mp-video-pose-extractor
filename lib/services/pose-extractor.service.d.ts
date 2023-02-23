import { EventEmitter } from '@angular/core';
import { Results } from '@mediapipe/holistic';
import * as i0 from "@angular/core";
export interface OnResultsEvent {
    mpResults: Results;
    frameImageDataUrl: string;
    posePreviewImageDataUrl: string;
    faceFrameImageDataUrl?: string;
}
/**
 * MediaPipe を用いて動画からポーズを抽出するためのサービス
 *
 * ※ シングルトンなサービスではないため、Component で providers に指定して使用することを想定
 */
export declare class PoseExtractorService {
    onResultsEventEmitter: EventEmitter<OnResultsEvent>;
    private holistic?;
    private posePreviewCanvasElement?;
    private posePreviewCanvasContext?;
    private handPreviewCanvasElement?;
    private handPreviewCanvasContext?;
    private facePreviewCanvasElement?;
    private facePreviewCanvasContext?;
    constructor();
    getPosePreviewMediaStream(): MediaStream | undefined;
    getHandPreviewMediaStream(): MediaStream | undefined;
    getFacePreviewMediaStream(): MediaStream | undefined;
    onVideoFrame(input: HTMLVideoElement | HTMLCanvasElement): Promise<void>;
    private init;
    private onResults;
    private connect;
    private removeElements;
    private getRectByLandmarks;
    static ɵfac: i0.ɵɵFactoryDeclaration<PoseExtractorService, never>;
    static ɵprov: i0.ɵɵInjectableDeclaration<PoseExtractorService>;
}
