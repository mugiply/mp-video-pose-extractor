import { Results } from '@mediapipe/holistic';
import { PoseSetItem } from '../interfaces/pose-set-item';
import { BodyVector } from '../interfaces/body-vector';
import { SimilarPoseItem } from '../interfaces/similar-pose-item';
import { HandVector } from '../interfaces/hand-vector';
export declare class PoseSet {
    generator?: string;
    version?: number;
    private videoMetadata;
    poses: PoseSetItem[];
    isFinalized?: boolean;
    static readonly BODY_VECTOR_MAPPINGS: string[];
    static readonly HAND_VECTOR_MAPPINGS: string[];
    private readonly IMAGE_WIDTH;
    private readonly IMAGE_MIME;
    private readonly IMAGE_QUALITY;
    private readonly IMAGE_MARGIN_TRIMMING_COLOR;
    private readonly IMAGE_MARGIN_TRIMMING_DIFF_THRESHOLD;
    private readonly IMAGE_BACKGROUND_REPLACE_SRC_COLOR;
    private readonly IMAGE_BACKGROUND_REPLACE_DST_COLOR;
    private readonly IMAGE_BACKGROUND_REPLACE_DIFF_THRESHOLD;
    constructor();
    getVideoName(): string;
    setVideoName(videoName: string): void;
    setVideoMetaData(width: number, height: number, duration: number): void;
    getNumberOfPoses(): number;
    getPoses(): PoseSetItem[];
    getPoseByTime(timeMiliseconds: number): PoseSetItem | undefined;
    pushPose(videoTimeMiliseconds: number, frameImageDataUrl: string | undefined, poseImageDataUrl: string | undefined, faceFrameImageDataUrl: string | undefined, results: Results): PoseSetItem | undefined;
    finalize(): Promise<void>;
    removeDuplicatedPoses(): void;
    getSimilarPoses(results: Results, threshold?: number): SimilarPoseItem[];
    static getBodyVector(poseLandmarks: {
        x: number;
        y: number;
        z: number;
    }[]): BodyVector | undefined;
    static getHandVectors(leftHandLandmarks: {
        x: number;
        y: number;
        z: number;
    }[], rightHandLandmarks: {
        x: number;
        y: number;
        z: number;
    }[]): HandVector | undefined;
    static isSimilarBodyPose(bodyVectorA: BodyVector, bodyVectorB: BodyVector, threshold?: number): boolean;
    static getBodyPoseSimilarity(bodyVectorA: BodyVector, bodyVectorB: BodyVector): number;
    static isSimilarHandPose(handVectorA: HandVector, handVectorB: HandVector, threshold?: number): boolean;
    static getHandSimilarity(handVectorA: HandVector, handVectorB: HandVector): number;
    getZip(): Promise<Blob>;
    getFileExtensionByMime(IMAGE_MIME: string): "png" | "jpg" | "webp";
    getJson(): Promise<string>;
    loadJson(json: string | any): void;
    loadZip(buffer: ArrayBuffer, includeImages?: boolean): Promise<void>;
}
