import { Results } from '@mediapipe/holistic';
import { PoseItem } from '../interfaces/pose-item';
import { PoseVector } from '../interfaces/pose-vector';
export declare class Pose {
    generator?: string;
    version?: number;
    private videoMetadata;
    poses: PoseItem[];
    isFinalized?: boolean;
    static readonly IS_ENABLE_DUPLICATED_POSE_REDUCTION = true;
    static readonly IS_SHRINK_RAW_POSE_DATA = false;
    static readonly POSE_VECTOR_MAPPINGS: string[];
    constructor();
    getVideoName(): string;
    setVideoName(videoName: string): void;
    setVideoMetaData(width: number, height: number, duration: number): void;
    getNumberOfPoses(): number;
    getPoses(): PoseItem[];
    pushPose(videoTimeMiliseconds: number, frameImageJpegDataUrl: string | undefined, videoWidth: number, videoHeight: number, videoDuration: number, results: Results): void;
    finalize(): void;
    getSimilarPoses(results: Results): PoseItem[];
    static getPoseVector(poseLandmarks: {
        x: number;
        y: number;
        z: number;
    }[]): PoseVector | undefined;
    static isSimilarPose(poseVectorA: PoseVector, poseVectorB: PoseVector, threshold?: number): boolean;
    getZip(): Promise<Blob>;
    getJson(): string;
    loadJson(json: string | any): void;
    loadZip(buffer: ArrayBuffer, includeImages?: boolean): Promise<void>;
}