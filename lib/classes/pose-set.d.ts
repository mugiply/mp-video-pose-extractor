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
    private similarPoseQueue;
    private readonly IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_WHOLE;
    private readonly IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_AROUND;
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
    /**
     * ポーズ数の取得
     * @returns
     */
    getNumberOfPoses(): number;
    /**
     * 全ポーズの取得
     * @returns 全てのポーズ
     */
    getPoses(): PoseSetItem[];
    /**
     * 指定された時間によるポーズの取得
     * @param timeMiliseconds ポーズの時間 (ミリ秒)
     * @returns ポーズ
     */
    getPoseByTime(timeMiliseconds: number): PoseSetItem | undefined;
    /**
     * ポーズの追加
     */
    pushPose(videoTimeMiliseconds: number, frameImageDataUrl: string | undefined, poseImageDataUrl: string | undefined, faceFrameImageDataUrl: string | undefined, results: Results): PoseSetItem | undefined;
    /**
     * ポーズの配列からポーズが決まっている瞬間を取得
     * @param poses ポーズの配列
     * @returns ポーズが決まっている瞬間
     */
    static getSuitablePoseByPoses(poses: PoseSetItem[]): PoseSetItem;
    /**
     * 最終処理
     * (重複したポーズの除去、画像のマージン除去など)
     */
    finalize(): Promise<void>;
    /**
     * 類似ポーズの取得
     * @param results MediaPipe Holistic によるポーズの検出結果
     * @param threshold しきい値
     * @param targetRange ポーズを比較する範囲 (all: 全て, bodyPose: 身体のみ, handPose: 手指のみ)
     * @returns 類似ポーズの配列
     */
    getSimilarPoses(results: Results, threshold?: number, targetRange?: 'all' | 'bodyPose' | 'handPose'): SimilarPoseItem[];
    /**
     * 身体の姿勢を表すベクトルの取得
     * @param poseLandmarks MediaPipe Holistic で取得できた身体のワールド座標 (ra 配列)
     * @returns ベクトル
     */
    static getBodyVector(poseLandmarks: {
        x: number;
        y: number;
        z: number;
    }[]): BodyVector | undefined;
    /**
     * 手指の姿勢を表すベクトルの取得
     * @param leftHandLandmarks MediaPipe Holistic で取得できた左手の正規化座標
     * @param rightHandLandmarks MediaPipe Holistic で取得できた右手の正規化座標
     * @returns ベクトル
     */
    static getHandVector(leftHandLandmarks: {
        x: number;
        y: number;
        z: number;
    }[], rightHandLandmarks: {
        x: number;
        y: number;
        z: number;
    }[]): HandVector | undefined;
    /**
     * BodyVector 間が類似しているかどうかの判定
     * @param bodyVectorA 比較先の BodyVector
     * @param bodyVectorB 比較元の BodyVector
     * @param threshold しきい値
     * @returns 類似しているかどうか
     */
    static isSimilarBodyPose(bodyVectorA: BodyVector, bodyVectorB: BodyVector, threshold?: number): boolean;
    /**
     * 身体ポーズの類似度の取得
     * @param bodyVectorA 比較先の BodyVector
     * @param bodyVectorB 比較元の BodyVector
     * @returns 類似度
     */
    static getBodyPoseSimilarity(bodyVectorA: BodyVector, bodyVectorB: BodyVector): number;
    /**
     * HandVector 間が類似しているかどうかの判定
     * @param handVectorA 比較先の HandVector
     * @param handVectorB 比較元の HandVector
     * @param threshold しきい値
     * @returns 類似しているかどうか
     */
    static isSimilarHandPose(handVectorA: HandVector, handVectorB: HandVector, threshold?: number): boolean;
    /**
     * 手のポーズの類似度の取得
     * @param handVectorA 比較先の HandVector
     * @param handVectorB 比較元の HandVector
     * @returns 類似度
     */
    static getHandSimilarity(handVectorA: HandVector, handVectorB: HandVector): number;
    /**
     * ZIP ファイルとしてのシリアライズ
     * @returns ZIPファイル (Blob 形式)
     */
    getZip(): Promise<Blob>;
    /**
     * JSON 文字列としてのシリアライズ
     * @returns JSON 文字列
     */
    getJson(): Promise<string>;
    /**
     * JSON からの読み込み
     * @param json JSON 文字列 または JSON オブジェクト
     */
    loadJson(json: string | any): void;
    /**
     * ZIP ファイルからの読み込み
     * @param buffer ZIP ファイルの Buffer
     * @param includeImages 画像を展開するかどうか
     */
    loadZip(buffer: ArrayBuffer, includeImages?: boolean): Promise<void>;
    private pushPoseFromSimilarPoseQueue;
    private removeDuplicatedPoses;
    private getFileExtensionByMime;
}
