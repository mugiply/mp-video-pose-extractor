import { POSE_LANDMARKS } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
// @ts-ignore
import cosSimilarity from 'cos-similarity';
import { ImageTrimmer } from './internals/image-trimmer';
export class PoseSet {
    constructor() {
        this.poses = [];
        this.isFinalized = false;
        // ポーズを追加するためのキュー
        this.similarPoseQueue = [];
        // 類似ポーズの除去 - 全ポーズから
        this.IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_WHOLE = false;
        // 類似ポーズの除去 - 各ポーズの前後から
        this.IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_AROUND = true;
        // 画像書き出し時の設定
        this.IMAGE_WIDTH = 1080;
        this.IMAGE_MIME = 'image/webp';
        this.IMAGE_QUALITY = 0.8;
        // 画像の余白除去
        this.IMAGE_MARGIN_TRIMMING_COLOR = '#000000';
        this.IMAGE_MARGIN_TRIMMING_DIFF_THRESHOLD = 50;
        // 画像の背景色置換
        this.IMAGE_BACKGROUND_REPLACE_SRC_COLOR = '#016AFD';
        this.IMAGE_BACKGROUND_REPLACE_DST_COLOR = '#FFFFFF00';
        this.IMAGE_BACKGROUND_REPLACE_DIFF_THRESHOLD = 130;
        this.videoMetadata = {
            name: '',
            width: 0,
            height: 0,
            duration: 0,
            firstPoseDetectedTime: 0,
        };
    }
    getVideoName() {
        return this.videoMetadata.name;
    }
    setVideoName(videoName) {
        this.videoMetadata.name = videoName;
    }
    setVideoMetaData(width, height, duration) {
        this.videoMetadata.width = width;
        this.videoMetadata.height = height;
        this.videoMetadata.duration = duration;
    }
    /**
     * ポーズ数の取得
     * @returns
     */
    getNumberOfPoses() {
        if (this.poses === undefined)
            return -1;
        return this.poses.length;
    }
    /**
     * 全ポーズの取得
     * @returns 全てのポーズ
     */
    getPoses() {
        if (this.poses === undefined)
            return [];
        return this.poses;
    }
    /**
     * 指定された時間によるポーズの取得
     * @param timeMiliseconds ポーズの時間 (ミリ秒)
     * @returns ポーズ
     */
    getPoseByTime(timeMiliseconds) {
        if (this.poses === undefined)
            return undefined;
        return this.poses.find((pose) => pose.timeMiliseconds === timeMiliseconds);
    }
    /**
     * ポーズの追加
     */
    pushPose(videoTimeMiliseconds, frameImageDataUrl, poseImageDataUrl, faceFrameImageDataUrl, results) {
        if (results.poseLandmarks === undefined)
            return;
        if (this.poses.length === 0) {
            this.videoMetadata.firstPoseDetectedTime = videoTimeMiliseconds;
        }
        const poseLandmarksWithWorldCoordinate = results.ea
            ? results.ea
            : [];
        if (poseLandmarksWithWorldCoordinate.length === 0) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the pose with the world coordinate`, results);
            return;
        }
        const bodyVector = PoseSet.getBodyVector(poseLandmarksWithWorldCoordinate);
        if (!bodyVector) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the body vector`, poseLandmarksWithWorldCoordinate);
            return;
        }
        if (results.leftHandLandmarks === undefined &&
            results.rightHandLandmarks === undefined) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the hand landmarks`, results);
        }
        else if (results.leftHandLandmarks === undefined) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the left hand landmarks`, results);
        }
        else if (results.rightHandLandmarks === undefined) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the right hand landmarks`, results);
        }
        const handVector = PoseSet.getHandVector(results.leftHandLandmarks, results.rightHandLandmarks);
        if (!handVector) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the hand vector`, results);
        }
        const pose = {
            timeMiliseconds: videoTimeMiliseconds,
            durationMiliseconds: -1,
            pose: poseLandmarksWithWorldCoordinate.map((worldCoordinateLandmark) => {
                return [
                    worldCoordinateLandmark.x,
                    worldCoordinateLandmark.y,
                    worldCoordinateLandmark.z,
                    worldCoordinateLandmark.visibility,
                ];
            }),
            leftHand: results.leftHandLandmarks?.map((normalizedLandmark) => {
                return [
                    normalizedLandmark.x,
                    normalizedLandmark.y,
                    normalizedLandmark.z,
                ];
            }),
            rightHand: results.rightHandLandmarks?.map((normalizedLandmark) => {
                return [
                    normalizedLandmark.x,
                    normalizedLandmark.y,
                    normalizedLandmark.z,
                ];
            }),
            bodyVector: bodyVector,
            handVector: handVector,
            frameImageDataUrl: frameImageDataUrl,
            poseImageDataUrl: poseImageDataUrl,
            faceFrameImageDataUrl: faceFrameImageDataUrl,
            extendedData: {},
            debug: {
                duplicatedItems: [],
            },
            mergedTimeMiliseconds: -1,
            mergedDurationMiliseconds: -1,
        };
        let lastPose;
        if (this.poses.length === 0 && 1 <= this.similarPoseQueue.length) {
            // 類似ポーズキューから最後のポーズを取得
            lastPose = this.similarPoseQueue[this.similarPoseQueue.length - 1];
        }
        else if (1 <= this.poses.length) {
            // ポーズ配列から最後のポーズを取得
            lastPose = this.poses[this.poses.length - 1];
        }
        if (lastPose) {
            // 最後のポーズがあれば、類似ポーズかどうかを比較
            const isSimilarBodyPose = PoseSet.isSimilarBodyPose(pose.bodyVector, lastPose.bodyVector);
            let isSimilarHandPose = true;
            if (lastPose.handVector && pose.handVector) {
                isSimilarHandPose = PoseSet.isSimilarHandPose(pose.handVector, lastPose.handVector);
            }
            else if (!lastPose.handVector && pose.handVector) {
                isSimilarHandPose = false;
            }
            if (!isSimilarBodyPose || !isSimilarHandPose) {
                // 身体・手のいずれかが前のポーズと類似していないならば、類似ポーズキューを処理して、ポーズ配列へ追加
                this.pushPoseFromSimilarPoseQueue(pose.timeMiliseconds);
            }
        }
        // 類似ポーズキューへ追加
        this.similarPoseQueue.push(pose);
        return pose;
    }
    /**
     * ポーズの配列からポーズが決まっている瞬間を取得
     * @param poses ポーズの配列
     * @returns ポーズが決まっている瞬間
     */
    static getSuitablePoseByPoses(poses) {
        if (poses.length === 0)
            return null;
        if (poses.length === 1) {
            return poses[1];
        }
        // 各標本ポーズごとの類似度を初期化
        const similaritiesOfPoses = {};
        for (let i = 0; i < poses.length; i++) {
            similaritiesOfPoses[poses[i].timeMiliseconds] = poses.map((pose) => {
                return {
                    handSimilarity: 0,
                    bodySimilarity: 0,
                };
            });
        }
        // 各標本ポーズごとの類似度を計算
        for (let samplePose of poses) {
            let handSimilarity;
            for (let i = 0; i < poses.length; i++) {
                const pose = poses[i];
                if (pose.handVector && samplePose.handVector) {
                    handSimilarity = PoseSet.getHandSimilarity(pose.handVector, samplePose.handVector);
                }
                let bodySimilarity = PoseSet.getBodyPoseSimilarity(pose.bodyVector, samplePose.bodyVector);
                similaritiesOfPoses[samplePose.timeMiliseconds][i] = {
                    handSimilarity: handSimilarity ?? 0,
                    bodySimilarity,
                };
            }
        }
        // 類似度の高いフレームが多かったポーズを選択
        const similaritiesOfSamplePoses = poses.map((pose) => {
            return similaritiesOfPoses[pose.timeMiliseconds].reduce((prev, current) => {
                return prev + current.handSimilarity + current.bodySimilarity;
            }, 0);
        });
        const maxSimilarity = Math.max(...similaritiesOfSamplePoses);
        const maxSimilarityIndex = similaritiesOfSamplePoses.indexOf(maxSimilarity);
        const selectedPose = poses[maxSimilarityIndex];
        if (!selectedPose) {
            console.warn(`[PoseSet] getSuitablePoseByPoses`, similaritiesOfSamplePoses, maxSimilarity, maxSimilarityIndex);
        }
        console.debug(`[PoseSet] getSuitablePoseByPoses`, {
            selected: selectedPose,
            unselected: poses.filter((pose) => {
                return pose.timeMiliseconds !== selectedPose.timeMiliseconds;
            }),
        });
        return selectedPose;
    }
    /**
     * 最終処理
     * (重複したポーズの除去、画像のマージン除去など)
     */
    async finalize() {
        if (this.similarPoseQueue.length > 0) {
            // 類似ポーズキューにポーズが残っている場合、最適なポーズを選択してポーズ配列へ追加
            this.pushPoseFromSimilarPoseQueue(this.videoMetadata.duration);
        }
        if (0 == this.poses.length) {
            // ポーズが一つもない場合、処理を終了
            this.isFinalized = true;
            return;
        }
        // ポーズの持続時間を設定
        for (let i = 0; i < this.poses.length - 1; i++) {
            if (this.poses[i].durationMiliseconds !== -1)
                continue;
            this.poses[i].durationMiliseconds =
                this.poses[i + 1].timeMiliseconds - this.poses[i].timeMiliseconds;
        }
        this.poses[this.poses.length - 1].durationMiliseconds =
            this.videoMetadata.duration -
                this.poses[this.poses.length - 1].timeMiliseconds;
        // 全体から重複ポーズを除去
        if (this.IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_WHOLE) {
            this.removeDuplicatedPoses();
        }
        // 最初のポーズを除去
        this.poses.shift();
        // 画像のマージンを取得
        console.debug(`[PoseSet] finalize - Detecting image margins...`);
        let imageTrimming = undefined;
        for (const pose of this.poses) {
            let imageTrimmer = new ImageTrimmer();
            if (!pose.frameImageDataUrl) {
                continue;
            }
            await imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);
            const marginColor = await imageTrimmer.getMarginColor();
            console.debug(`[PoseSet] finalize - Detected margin color...`, pose.timeMiliseconds, marginColor);
            if (marginColor === null)
                continue;
            if (marginColor !== this.IMAGE_MARGIN_TRIMMING_COLOR) {
                continue;
            }
            const trimmed = await imageTrimmer.trimMargin(marginColor, this.IMAGE_MARGIN_TRIMMING_DIFF_THRESHOLD);
            if (!trimmed)
                continue;
            imageTrimming = trimmed;
            console.debug(`[PoseSet] finalize - Determined image trimming positions...`, trimmed);
            break;
        }
        // 画像を整形
        for (const pose of this.poses) {
            let imageTrimmer = new ImageTrimmer();
            if (!pose.frameImageDataUrl || !pose.poseImageDataUrl) {
                continue;
            }
            console.debug(`[PoseSet] finalize - Processing image...`, pose.timeMiliseconds);
            // 画像を整形 - フレーム画像
            await imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);
            if (imageTrimming) {
                await imageTrimmer.crop(0, imageTrimming.marginTop, imageTrimming.width, imageTrimming.heightNew);
            }
            await imageTrimmer.replaceColor(this.IMAGE_BACKGROUND_REPLACE_SRC_COLOR, this.IMAGE_BACKGROUND_REPLACE_DST_COLOR, this.IMAGE_BACKGROUND_REPLACE_DIFF_THRESHOLD);
            await imageTrimmer.resizeWithFit({
                width: this.IMAGE_WIDTH,
            });
            let newDataUrl = await imageTrimmer.getDataUrl(this.IMAGE_MIME, this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
                ? this.IMAGE_QUALITY
                : undefined);
            if (!newDataUrl) {
                console.warn(`[PoseSet] finalize - Could not get the new dataurl for frame image`);
                continue;
            }
            pose.frameImageDataUrl = newDataUrl;
            // 画像を整形 - ポーズプレビュー画像
            imageTrimmer = new ImageTrimmer();
            await imageTrimmer.loadByDataUrl(pose.poseImageDataUrl);
            if (imageTrimming) {
                await imageTrimmer.crop(0, imageTrimming.marginTop, imageTrimming.width, imageTrimming.heightNew);
            }
            await imageTrimmer.resizeWithFit({
                width: this.IMAGE_WIDTH,
            });
            newDataUrl = await imageTrimmer.getDataUrl(this.IMAGE_MIME, this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
                ? this.IMAGE_QUALITY
                : undefined);
            if (!newDataUrl) {
                console.warn(`[PoseSet] finalize - Could not get the new dataurl for pose preview image`);
                continue;
            }
            pose.poseImageDataUrl = newDataUrl;
            if (pose.faceFrameImageDataUrl) {
                // 画像を整形 - 顔フレーム画像
                imageTrimmer = new ImageTrimmer();
                await imageTrimmer.loadByDataUrl(pose.faceFrameImageDataUrl);
                newDataUrl = await imageTrimmer.getDataUrl(this.IMAGE_MIME, this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
                    ? this.IMAGE_QUALITY
                    : undefined);
                if (!newDataUrl) {
                    console.warn(`[PoseSet] finalize - Could not get the new dataurl for face frame image`);
                    continue;
                }
                pose.faceFrameImageDataUrl = newDataUrl;
            }
        }
        this.isFinalized = true;
    }
    /**
     * 類似ポーズの取得
     * @param results MediaPipe Holistic によるポーズの検出結果
     * @param threshold しきい値
     * @param targetRange ポーズを比較する範囲 (all: 全て, bodyPose: 身体のみ, handPose: 手指のみ)
     * @returns 類似ポーズの配列
     */
    getSimilarPoses(results, threshold = 0.9, targetRange = 'all') {
        // 身体のベクトルを取得
        let bodyVector;
        try {
            bodyVector = PoseSet.getBodyVector(results.ea);
        }
        catch (e) {
            console.error(`[PoseSet] getSimilarPoses - Error occurred`, e, results);
            return [];
        }
        if (!bodyVector) {
            throw 'Could not get the body vector';
        }
        // 手指のベクトルを取得
        let handVector;
        if (targetRange === 'all' || targetRange === 'handPose') {
            handVector = PoseSet.getHandVector(results.leftHandLandmarks, results.rightHandLandmarks);
            if (targetRange === 'handPose' && !handVector) {
                throw 'Could not get the hand vector';
            }
        }
        // 各ポーズとベクトルを比較
        const poses = [];
        for (const pose of this.poses) {
            if ((targetRange === 'all' || targetRange === 'bodyPose') &&
                !pose.bodyVector) {
                continue;
            }
            else if (targetRange === 'handPose' && !pose.handVector) {
                continue;
            }
            /*console.debug(
              '[PoseSet] getSimilarPoses - ',
              this.getVideoName(),
              pose.timeMiliseconds
            );*/
            // 身体のポーズの類似度を取得
            let bodySimilarity;
            if (bodyVector && pose.bodyVector) {
                bodySimilarity = PoseSet.getBodyPoseSimilarity(pose.bodyVector, bodyVector);
            }
            // 手指のポーズの類似度を取得
            let handSimilarity;
            if (handVector && pose.handVector) {
                handSimilarity = PoseSet.getHandSimilarity(pose.handVector, handVector);
            }
            // 判定
            let similarity, isSimilar = false;
            if (targetRange === 'all') {
                similarity = Math.max(bodySimilarity ?? 0, handSimilarity ?? 0);
                if (threshold <= bodySimilarity || threshold <= handSimilarity) {
                    isSimilar = true;
                }
            }
            else if (targetRange === 'bodyPose') {
                similarity = bodySimilarity;
                if (threshold <= bodySimilarity) {
                    isSimilar = true;
                }
            }
            else if (targetRange === 'handPose') {
                similarity = handSimilarity;
                if (threshold <= handSimilarity) {
                    isSimilar = true;
                }
            }
            if (!isSimilar)
                continue;
            // 結果へ追加
            poses.push({
                ...pose,
                similarity: similarity,
                bodyPoseSimilarity: bodySimilarity,
                handPoseSimilarity: handSimilarity,
            });
        }
        return poses;
    }
    /**
     * 身体の姿勢を表すベクトルの取得
     * @param poseLandmarks MediaPipe Holistic で取得できた身体のワールド座標 (ra 配列)
     * @returns ベクトル
     */
    static getBodyVector(poseLandmarks) {
        return {
            rightWristToRightElbow: [
                poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].x -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].x,
                poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].y -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].y,
                poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].z -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].z,
            ],
            rightElbowToRightShoulder: [
                poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].x -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER].x,
                poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].y -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER].y,
                poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].z -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER].z,
            ],
            leftWristToLeftElbow: [
                poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].x -
                    poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].x,
                poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].y -
                    poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].y,
                poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].z -
                    poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].z,
            ],
            leftElbowToLeftShoulder: [
                poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].x -
                    poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER].x,
                poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].y -
                    poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER].y,
                poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].z -
                    poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER].z,
            ],
        };
    }
    /**
     * 手指の姿勢を表すベクトルの取得
     * @param leftHandLandmarks MediaPipe Holistic で取得できた左手の正規化座標
     * @param rightHandLandmarks MediaPipe Holistic で取得できた右手の正規化座標
     * @returns ベクトル
     */
    static getHandVector(leftHandLandmarks, rightHandLandmarks) {
        if ((rightHandLandmarks === undefined || rightHandLandmarks.length === 0) &&
            (leftHandLandmarks === undefined || leftHandLandmarks.length === 0)) {
            return undefined;
        }
        return {
            // 右手 - 親指
            rightThumbTipToFirstJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[4].x - rightHandLandmarks[3].x,
                    rightHandLandmarks[4].y - rightHandLandmarks[3].y,
                    rightHandLandmarks[4].z - rightHandLandmarks[3].z,
                ],
            rightThumbFirstJointToSecondJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[3].x - rightHandLandmarks[2].x,
                    rightHandLandmarks[3].y - rightHandLandmarks[2].y,
                    rightHandLandmarks[3].z - rightHandLandmarks[2].z,
                ],
            // 右手 - 人差し指
            rightIndexFingerTipToFirstJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[8].x - rightHandLandmarks[7].x,
                    rightHandLandmarks[8].y - rightHandLandmarks[7].y,
                    rightHandLandmarks[8].z - rightHandLandmarks[7].z,
                ],
            rightIndexFingerFirstJointToSecondJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[7].x - rightHandLandmarks[6].x,
                    rightHandLandmarks[7].y - rightHandLandmarks[6].y,
                    rightHandLandmarks[7].z - rightHandLandmarks[6].z,
                ],
            // 右手 - 中指
            rightMiddleFingerTipToFirstJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[12].x - rightHandLandmarks[11].x,
                    rightHandLandmarks[12].y - rightHandLandmarks[11].y,
                    rightHandLandmarks[12].z - rightHandLandmarks[11].z,
                ],
            rightMiddleFingerFirstJointToSecondJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[11].x - rightHandLandmarks[10].x,
                    rightHandLandmarks[11].y - rightHandLandmarks[10].y,
                    rightHandLandmarks[11].z - rightHandLandmarks[10].z,
                ],
            // 右手 - 薬指
            rightRingFingerTipToFirstJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[16].x - rightHandLandmarks[15].x,
                    rightHandLandmarks[16].y - rightHandLandmarks[15].y,
                    rightHandLandmarks[16].z - rightHandLandmarks[15].z,
                ],
            rightRingFingerFirstJointToSecondJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[15].x - rightHandLandmarks[14].x,
                    rightHandLandmarks[15].y - rightHandLandmarks[14].y,
                    rightHandLandmarks[15].z - rightHandLandmarks[14].z,
                ],
            // 右手 - 小指
            rightPinkyFingerTipToFirstJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[20].x - rightHandLandmarks[19].x,
                    rightHandLandmarks[20].y - rightHandLandmarks[19].y,
                    rightHandLandmarks[20].z - rightHandLandmarks[19].z,
                ],
            rightPinkyFingerFirstJointToSecondJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[19].x - rightHandLandmarks[18].x,
                    rightHandLandmarks[19].y - rightHandLandmarks[18].y,
                    rightHandLandmarks[19].z - rightHandLandmarks[18].z,
                ],
            // 左手 - 親指
            leftThumbTipToFirstJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[4].x - leftHandLandmarks[3].x,
                    leftHandLandmarks[4].y - leftHandLandmarks[3].y,
                    leftHandLandmarks[4].z - leftHandLandmarks[3].z,
                ],
            leftThumbFirstJointToSecondJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[3].x - leftHandLandmarks[2].x,
                    leftHandLandmarks[3].y - leftHandLandmarks[2].y,
                    leftHandLandmarks[3].z - leftHandLandmarks[2].z,
                ],
            // 左手 - 人差し指
            leftIndexFingerTipToFirstJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[8].x - leftHandLandmarks[7].x,
                    leftHandLandmarks[8].y - leftHandLandmarks[7].y,
                    leftHandLandmarks[8].z - leftHandLandmarks[7].z,
                ],
            leftIndexFingerFirstJointToSecondJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[7].x - leftHandLandmarks[6].x,
                    leftHandLandmarks[7].y - leftHandLandmarks[6].y,
                    leftHandLandmarks[7].z - leftHandLandmarks[6].z,
                ],
            // 左手 - 中指
            leftMiddleFingerTipToFirstJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[12].x - leftHandLandmarks[11].x,
                    leftHandLandmarks[12].y - leftHandLandmarks[11].y,
                    leftHandLandmarks[12].z - leftHandLandmarks[11].z,
                ],
            leftMiddleFingerFirstJointToSecondJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[11].x - leftHandLandmarks[10].x,
                    leftHandLandmarks[11].y - leftHandLandmarks[10].y,
                    leftHandLandmarks[11].z - leftHandLandmarks[10].z,
                ],
            // 左手 - 薬指
            leftRingFingerTipToFirstJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[16].x - leftHandLandmarks[15].x,
                    leftHandLandmarks[16].y - leftHandLandmarks[15].y,
                    leftHandLandmarks[16].z - leftHandLandmarks[15].z,
                ],
            leftRingFingerFirstJointToSecondJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[15].x - leftHandLandmarks[14].x,
                    leftHandLandmarks[15].y - leftHandLandmarks[14].y,
                    leftHandLandmarks[15].z - leftHandLandmarks[14].z,
                ],
            // 左手 - 小指
            leftPinkyFingerTipToFirstJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[20].x - leftHandLandmarks[19].x,
                    leftHandLandmarks[20].y - leftHandLandmarks[19].y,
                    leftHandLandmarks[20].z - leftHandLandmarks[19].z,
                ],
            leftPinkyFingerFirstJointToSecondJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[19].x - leftHandLandmarks[18].x,
                    leftHandLandmarks[19].y - leftHandLandmarks[18].y,
                    leftHandLandmarks[19].z - leftHandLandmarks[18].z,
                ],
        };
    }
    /**
     * BodyVector 間が類似しているかどうかの判定
     * @param bodyVectorA 比較先の BodyVector
     * @param bodyVectorB 比較元の BodyVector
     * @param threshold しきい値
     * @returns 類似しているかどうか
     */
    static isSimilarBodyPose(bodyVectorA, bodyVectorB, threshold = 0.8) {
        let isSimilar = false;
        const similarity = PoseSet.getBodyPoseSimilarity(bodyVectorA, bodyVectorB);
        if (similarity >= threshold)
            isSimilar = true;
        // console.debug(`[PoseSet] isSimilarPose`, isSimilar, similarity);
        return isSimilar;
    }
    /**
     * 身体ポーズの類似度の取得
     * @param bodyVectorA 比較先の BodyVector
     * @param bodyVectorB 比較元の BodyVector
     * @returns 類似度
     */
    static getBodyPoseSimilarity(bodyVectorA, bodyVectorB) {
        const cosSimilarities = {
            leftWristToLeftElbow: cosSimilarity(bodyVectorA.leftWristToLeftElbow, bodyVectorB.leftWristToLeftElbow),
            leftElbowToLeftShoulder: cosSimilarity(bodyVectorA.leftElbowToLeftShoulder, bodyVectorB.leftElbowToLeftShoulder),
            rightWristToRightElbow: cosSimilarity(bodyVectorA.rightWristToRightElbow, bodyVectorB.rightWristToRightElbow),
            rightElbowToRightShoulder: cosSimilarity(bodyVectorA.rightElbowToRightShoulder, bodyVectorB.rightElbowToRightShoulder),
        };
        const cosSimilaritiesSum = Object.values(cosSimilarities).reduce((sum, value) => sum + value, 0);
        return cosSimilaritiesSum / Object.keys(cosSimilarities).length;
    }
    /**
     * HandVector 間が類似しているかどうかの判定
     * @param handVectorA 比較先の HandVector
     * @param handVectorB 比較元の HandVector
     * @param threshold しきい値
     * @returns 類似しているかどうか
     */
    static isSimilarHandPose(handVectorA, handVectorB, threshold = 0.75) {
        const similarity = PoseSet.getHandSimilarity(handVectorA, handVectorB);
        if (similarity === -1) {
            return true;
        }
        return similarity >= threshold;
    }
    /**
     * 手のポーズの類似度の取得
     * @param handVectorA 比較先の HandVector
     * @param handVectorB 比較元の HandVector
     * @returns 類似度
     */
    static getHandSimilarity(handVectorA, handVectorB) {
        const cosSimilaritiesRightHand = handVectorA.rightThumbFirstJointToSecondJoint === null ||
            handVectorB.rightThumbFirstJointToSecondJoint === null
            ? undefined
            : {
                // 右手 - 親指
                rightThumbTipToFirstJoint: cosSimilarity(handVectorA.rightThumbTipToFirstJoint, handVectorB.rightThumbTipToFirstJoint),
                rightThumbFirstJointToSecondJoint: cosSimilarity(handVectorA.rightThumbFirstJointToSecondJoint, handVectorB.rightThumbFirstJointToSecondJoint),
                // 右手 - 人差し指
                rightIndexFingerTipToFirstJoint: cosSimilarity(handVectorA.rightIndexFingerTipToFirstJoint, handVectorB.rightIndexFingerTipToFirstJoint),
                rightIndexFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.rightIndexFingerFirstJointToSecondJoint, handVectorB.rightIndexFingerFirstJointToSecondJoint),
                // 右手 - 中指
                rightMiddleFingerTipToFirstJoint: cosSimilarity(handVectorA.rightMiddleFingerTipToFirstJoint, handVectorB.rightMiddleFingerTipToFirstJoint),
                rightMiddleFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.rightMiddleFingerFirstJointToSecondJoint, handVectorB.rightMiddleFingerFirstJointToSecondJoint),
                // 右手 - 薬指
                rightRingFingerTipToFirstJoint: cosSimilarity(handVectorA.rightRingFingerTipToFirstJoint, handVectorB.rightRingFingerFirstJointToSecondJoint),
                rightRingFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.rightRingFingerFirstJointToSecondJoint, handVectorB.rightRingFingerFirstJointToSecondJoint),
                // 右手 - 小指
                rightPinkyFingerTipToFirstJoint: cosSimilarity(handVectorA.rightPinkyFingerTipToFirstJoint, handVectorB.rightPinkyFingerTipToFirstJoint),
                rightPinkyFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.rightPinkyFingerFirstJointToSecondJoint, handVectorB.rightPinkyFingerFirstJointToSecondJoint),
            };
        const cosSimilaritiesLeftHand = handVectorA.leftThumbFirstJointToSecondJoint === null ||
            handVectorB.leftThumbFirstJointToSecondJoint === null
            ? undefined
            : {
                // 左手 - 親指
                leftThumbTipToFirstJoint: cosSimilarity(handVectorA.leftThumbTipToFirstJoint, handVectorB.leftThumbTipToFirstJoint),
                leftThumbFirstJointToSecondJoint: cosSimilarity(handVectorA.leftThumbFirstJointToSecondJoint, handVectorB.leftThumbFirstJointToSecondJoint),
                // 左手 - 人差し指
                leftIndexFingerTipToFirstJoint: cosSimilarity(handVectorA.leftIndexFingerTipToFirstJoint, handVectorB.leftIndexFingerTipToFirstJoint),
                leftIndexFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.leftIndexFingerFirstJointToSecondJoint, handVectorB.leftIndexFingerFirstJointToSecondJoint),
                // 左手 - 中指
                leftMiddleFingerTipToFirstJoint: cosSimilarity(handVectorA.leftMiddleFingerTipToFirstJoint, handVectorB.leftMiddleFingerTipToFirstJoint),
                leftMiddleFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.leftMiddleFingerFirstJointToSecondJoint, handVectorB.leftMiddleFingerFirstJointToSecondJoint),
                // 左手 - 薬指
                leftRingFingerTipToFirstJoint: cosSimilarity(handVectorA.leftRingFingerTipToFirstJoint, handVectorB.leftRingFingerTipToFirstJoint),
                leftRingFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.leftRingFingerFirstJointToSecondJoint, handVectorB.leftRingFingerFirstJointToSecondJoint),
                // 左手 - 小指
                leftPinkyFingerTipToFirstJoint: cosSimilarity(handVectorA.leftPinkyFingerTipToFirstJoint, handVectorB.leftPinkyFingerTipToFirstJoint),
                leftPinkyFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.leftPinkyFingerFirstJointToSecondJoint, handVectorB.leftPinkyFingerFirstJointToSecondJoint),
            };
        // 左手の類似度
        let cosSimilaritiesSumLeftHand = 0;
        if (cosSimilaritiesLeftHand) {
            cosSimilaritiesSumLeftHand = Object.values(cosSimilaritiesLeftHand).reduce((sum, value) => sum + value, 0);
        }
        // 右手の類似度
        let cosSimilaritiesSumRightHand = 0;
        if (cosSimilaritiesRightHand) {
            cosSimilaritiesSumRightHand = Object.values(cosSimilaritiesRightHand).reduce((sum, value) => sum + value, 0);
        }
        // 合算された類似度
        if (cosSimilaritiesRightHand && cosSimilaritiesLeftHand) {
            return ((cosSimilaritiesSumRightHand + cosSimilaritiesSumLeftHand) /
                (Object.keys(cosSimilaritiesRightHand).length +
                    Object.keys(cosSimilaritiesLeftHand).length));
        }
        else if (cosSimilaritiesRightHand) {
            if (handVectorB.leftThumbFirstJointToSecondJoint !== null &&
                handVectorA.leftThumbFirstJointToSecondJoint === null) {
                // handVectorB で左手があるのに handVectorA で左手がない場合、類似度を減らす
                console.debug(`[PoseSet] getHandSimilarity - Adjust similarity, because left hand not found...`);
                return (cosSimilaritiesSumRightHand /
                    (Object.keys(cosSimilaritiesRightHand).length * 2));
            }
            return (cosSimilaritiesSumRightHand /
                Object.keys(cosSimilaritiesRightHand).length);
        }
        else if (cosSimilaritiesLeftHand) {
            if (handVectorB.rightThumbFirstJointToSecondJoint !== null &&
                handVectorA.rightThumbFirstJointToSecondJoint === null) {
                // handVectorB で右手があるのに handVectorA で右手がない場合、類似度を減らす
                console.debug(`[PoseSet] getHandSimilarity - Adjust similarity, because right hand not found...`);
                return (cosSimilaritiesSumLeftHand /
                    (Object.keys(cosSimilaritiesLeftHand).length * 2));
            }
            return (cosSimilaritiesSumLeftHand /
                Object.keys(cosSimilaritiesLeftHand).length);
        }
        return -1;
    }
    /**
     * ZIP ファイルとしてのシリアライズ
     * @returns ZIPファイル (Blob 形式)
     */
    async getZip() {
        const jsZip = new JSZip();
        jsZip.file('poses.json', await this.getJson());
        const imageFileExt = this.getFileExtensionByMime(this.IMAGE_MIME);
        for (const pose of this.poses) {
            if (pose.frameImageDataUrl) {
                try {
                    const index = pose.frameImageDataUrl.indexOf('base64,') + 'base64,'.length;
                    const base64 = pose.frameImageDataUrl.substring(index);
                    jsZip.file(`frame-${pose.timeMiliseconds}.${imageFileExt}`, base64, {
                        base64: true,
                    });
                }
                catch (error) {
                    console.warn(`[PoseExporterService] push - Could not push frame image`, error);
                    throw error;
                }
            }
            if (pose.poseImageDataUrl) {
                try {
                    const index = pose.poseImageDataUrl.indexOf('base64,') + 'base64,'.length;
                    const base64 = pose.poseImageDataUrl.substring(index);
                    jsZip.file(`pose-${pose.timeMiliseconds}.${imageFileExt}`, base64, {
                        base64: true,
                    });
                }
                catch (error) {
                    console.warn(`[PoseExporterService] push - Could not push frame image`, error);
                    throw error;
                }
            }
            if (pose.faceFrameImageDataUrl) {
                try {
                    const index = pose.faceFrameImageDataUrl.indexOf('base64,') + 'base64,'.length;
                    const base64 = pose.faceFrameImageDataUrl.substring(index);
                    jsZip.file(`face-${pose.timeMiliseconds}.${imageFileExt}`, base64, {
                        base64: true,
                    });
                }
                catch (error) {
                    console.warn(`[PoseExporterService] push - Could not push face frame image`, error);
                    throw error;
                }
            }
        }
        return await jsZip.generateAsync({ type: 'blob' });
    }
    /**
     * JSON 文字列としてのシリアライズ
     * @returns JSON 文字列
     */
    async getJson() {
        if (this.videoMetadata === undefined || this.poses === undefined)
            return '{}';
        if (!this.isFinalized) {
            await this.finalize();
        }
        let poseLandmarkMappings = [];
        for (const key of Object.keys(POSE_LANDMARKS)) {
            const index = POSE_LANDMARKS[key];
            poseLandmarkMappings[index] = key;
        }
        const json = {
            generator: 'mp-video-pose-extractor',
            version: 1,
            video: this.videoMetadata,
            poses: this.poses.map((pose) => {
                // BodyVector の圧縮
                const bodyVector = [];
                for (const key of PoseSet.BODY_VECTOR_MAPPINGS) {
                    bodyVector.push(pose.bodyVector[key]);
                }
                // HandVector の圧縮
                let handVector = undefined;
                if (pose.handVector) {
                    handVector = [];
                    for (const key of PoseSet.HAND_VECTOR_MAPPINGS) {
                        handVector.push(pose.handVector[key]);
                    }
                }
                // PoseSetJsonItem の pose オブジェクトを生成
                return {
                    t: pose.timeMiliseconds,
                    d: pose.durationMiliseconds,
                    p: pose.pose,
                    l: pose.leftHand,
                    r: pose.rightHand,
                    v: bodyVector,
                    h: handVector,
                    e: pose.extendedData,
                    md: pose.mergedDurationMiliseconds,
                    mt: pose.mergedTimeMiliseconds,
                };
            }),
            poseLandmarkMapppings: poseLandmarkMappings,
        };
        return JSON.stringify(json);
    }
    /**
     * JSON からの読み込み
     * @param json JSON 文字列 または JSON オブジェクト
     */
    loadJson(json) {
        const parsedJson = typeof json === 'string' ? JSON.parse(json) : json;
        if (parsedJson.generator !== 'mp-video-pose-extractor') {
            throw '不正なファイル';
        }
        else if (parsedJson.version !== 1) {
            throw '未対応のバージョン';
        }
        this.videoMetadata = parsedJson.video;
        this.poses = parsedJson.poses.map((item) => {
            const bodyVector = {};
            PoseSet.BODY_VECTOR_MAPPINGS.map((key, index) => {
                bodyVector[key] = item.v[index];
            });
            const handVector = {};
            if (item.h) {
                PoseSet.HAND_VECTOR_MAPPINGS.map((key, index) => {
                    handVector[key] = item.h[index];
                });
            }
            return {
                timeMiliseconds: item.t,
                durationMiliseconds: item.d,
                pose: item.p,
                leftHand: item.l,
                rightHand: item.r,
                bodyVector: bodyVector,
                handVector: handVector,
                frameImageDataUrl: undefined,
                extendedData: item.e,
                debug: undefined,
                mergedDurationMiliseconds: item.md,
                mergedTimeMiliseconds: item.mt,
            };
        });
    }
    /**
     * ZIP ファイルからの読み込み
     * @param buffer ZIP ファイルの Buffer
     * @param includeImages 画像を展開するかどうか
     */
    async loadZip(buffer, includeImages = true) {
        const jsZip = new JSZip();
        console.debug(`[PoseSet] init...`);
        const zip = await jsZip.loadAsync(buffer, { base64: false });
        if (!zip)
            throw 'ZIPファイルを読み込めませんでした';
        const json = await zip.file('poses.json')?.async('text');
        if (json === undefined) {
            throw 'ZIPファイルに pose.json が含まれていません';
        }
        this.loadJson(json);
        const fileExt = this.getFileExtensionByMime(this.IMAGE_MIME);
        if (includeImages) {
            for (const pose of this.poses) {
                if (!pose.frameImageDataUrl) {
                    const frameImageFileName = `frame-${pose.timeMiliseconds}.${fileExt}`;
                    const imageBase64 = await zip
                        .file(frameImageFileName)
                        ?.async('base64');
                    if (imageBase64) {
                        pose.frameImageDataUrl = `data:${this.IMAGE_MIME};base64,${imageBase64}`;
                    }
                }
                if (!pose.poseImageDataUrl) {
                    const poseImageFileName = `pose-${pose.timeMiliseconds}.${fileExt}`;
                    const imageBase64 = await zip
                        .file(poseImageFileName)
                        ?.async('base64');
                    if (imageBase64) {
                        pose.poseImageDataUrl = `data:${this.IMAGE_MIME};base64,${imageBase64}`;
                    }
                }
            }
        }
    }
    pushPoseFromSimilarPoseQueue(nextPoseTimeMiliseconds) {
        if (this.similarPoseQueue.length === 0)
            return;
        if (this.similarPoseQueue.length === 1) {
            // 類似ポーズキューにポーズが一つしかない場合、当該ポーズをポーズ配列へ追加
            const pose = this.similarPoseQueue[0];
            this.poses.push(pose);
            this.similarPoseQueue = [];
            return;
        }
        // 各ポーズの持続時間を設定
        for (let i = 0; i < this.similarPoseQueue.length - 1; i++) {
            this.similarPoseQueue[i].durationMiliseconds =
                this.similarPoseQueue[i + 1].timeMiliseconds -
                    this.similarPoseQueue[i].timeMiliseconds;
        }
        if (nextPoseTimeMiliseconds) {
            this.similarPoseQueue[this.similarPoseQueue.length - 1].durationMiliseconds =
                nextPoseTimeMiliseconds -
                    this.similarPoseQueue[this.similarPoseQueue.length - 1].timeMiliseconds;
        }
        // 類似ポーズキューの中から最も持続時間が長いポーズを選択
        const selectedPose = PoseSet.getSuitablePoseByPoses(this.similarPoseQueue);
        // 選択されなかったポーズを列挙
        selectedPose.debug.duplicatedItems = this.similarPoseQueue
            .filter((item) => {
            return item.timeMiliseconds !== selectedPose.timeMiliseconds;
        })
            .map((item) => {
            return {
                timeMiliseconds: item.timeMiliseconds,
                durationMiliseconds: item.durationMiliseconds,
                bodySimilarity: undefined,
                handSimilarity: undefined,
            };
        });
        selectedPose.mergedTimeMiliseconds =
            this.similarPoseQueue[0].timeMiliseconds;
        selectedPose.mergedDurationMiliseconds = this.similarPoseQueue.reduce((sum, item) => {
            return sum + item.durationMiliseconds;
        }, 0);
        // 当該ポーズをポーズ配列へ追加
        if (this.IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_AROUND) {
            this.poses.push(selectedPose);
        }
        else {
            // デバッグ用
            this.poses.push(...this.similarPoseQueue);
        }
        // 類似ポーズキューをクリア
        this.similarPoseQueue = [];
    }
    removeDuplicatedPoses() {
        // 全ポーズを比較して類似ポーズを削除
        const newPoses = [], removedPoses = [];
        for (const pose of this.poses) {
            let duplicatedPose;
            for (const insertedPose of newPoses) {
                const isSimilarBodyPose = PoseSet.isSimilarBodyPose(pose.bodyVector, insertedPose.bodyVector);
                const isSimilarHandPose = pose.handVector && insertedPose.handVector
                    ? PoseSet.isSimilarHandPose(pose.handVector, insertedPose.handVector, 0.9)
                    : false;
                if (isSimilarBodyPose && isSimilarHandPose) {
                    // 身体・手ともに類似ポーズならば
                    duplicatedPose = insertedPose;
                    break;
                }
            }
            if (duplicatedPose) {
                removedPoses.push(pose);
                if (duplicatedPose.debug.duplicatedItems) {
                    duplicatedPose.debug.duplicatedItems.push({
                        timeMiliseconds: pose.timeMiliseconds,
                        durationMiliseconds: pose.durationMiliseconds,
                        bodySimilarity: undefined,
                        handSimilarity: undefined,
                    });
                }
                continue;
            }
            newPoses.push(pose);
        }
        console.info(`[PoseSet] removeDuplicatedPoses - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`, {
            removed: removedPoses,
            keeped: newPoses,
        });
        this.poses = newPoses;
    }
    getFileExtensionByMime(IMAGE_MIME) {
        switch (IMAGE_MIME) {
            case 'image/png':
                return 'png';
            case 'image/jpeg':
                return 'jpg';
            case 'image/webp':
                return 'webp';
            default:
                return 'png';
        }
    }
}
// BodyVector のキー名
PoseSet.BODY_VECTOR_MAPPINGS = [
    // 右腕
    'rightWristToRightElbow',
    'rightElbowToRightShoulder',
    // 左腕
    'leftWristToLeftElbow',
    'leftElbowToLeftShoulder',
];
// HandVector のキー名
PoseSet.HAND_VECTOR_MAPPINGS = [
    // 右手 - 親指
    'rightThumbTipToFirstJoint',
    'rightThumbFirstJointToSecondJoint',
    // 右手 - 人差し指
    'rightIndexFingerTipToFirstJoint',
    'rightIndexFingerFirstJointToSecondJoint',
    // 右手 - 中指
    'rightMiddleFingerTipToFirstJoint',
    'rightMiddleFingerFirstJointToSecondJoint',
    // 右手 - 薬指
    'rightRingFingerTipToFirstJoint',
    'rightRingFingerFirstJointToSecondJoint',
    // 右手 - 小指
    'rightPinkyFingerTipToFirstJoint',
    'rightPinkyFingerFirstJointToSecondJoint',
    // 左手 - 親指
    'leftThumbTipToFirstJoint',
    'leftThumbFirstJointToSecondJoint',
    // 左手 - 人差し指
    'leftIndexFingerTipToFirstJoint',
    'leftIndexFingerFirstJointToSecondJoint',
    // 左手 - 中指
    'leftMiddleFingerTipToFirstJoint',
    'leftMiddleFingerFirstJointToSecondJoint',
    // 左手 - 薬指
    'leftRingFingerTipToFirstJoint',
    'leftRingFingerFirstJointToSecondJoint',
    // 左手 - 小指
    'leftPinkyFingerTipToFirstJoint',
    'leftPinkyFingerFirstJointToSecondJoint',
];
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1zZXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi9wcm9qZWN0cy9uZ3gtbXAtcG9zZS1leHRyYWN0b3Ivc3JjL2xpYi9jbGFzc2VzL3Bvc2Utc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxjQUFjLEVBQVcsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssS0FBSyxNQUFNLE9BQU8sQ0FBQztBQU0vQixhQUFhO0FBQ2IsT0FBTyxhQUFhLE1BQU0sZ0JBQWdCLENBQUM7QUFFM0MsT0FBTyxFQUFFLFlBQVksRUFBRSxNQUFNLDJCQUEyQixDQUFDO0FBR3pELE1BQU0sT0FBTyxPQUFPO0lBZ0ZsQjtRQXRFTyxVQUFLLEdBQWtCLEVBQUUsQ0FBQztRQUMxQixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQThDckMsaUJBQWlCO1FBQ1QscUJBQWdCLEdBQWtCLEVBQUUsQ0FBQztRQUU3QyxvQkFBb0I7UUFDSCxpREFBNEMsR0FBRyxLQUFLLENBQUM7UUFDdEUsdUJBQXVCO1FBQ04sa0RBQTZDLEdBQUcsSUFBSSxDQUFDO1FBRXRFLGFBQWE7UUFDSSxnQkFBVyxHQUFXLElBQUksQ0FBQztRQUMzQixlQUFVLEdBQ3pCLFlBQVksQ0FBQztRQUNFLGtCQUFhLEdBQUcsR0FBRyxDQUFDO1FBRXJDLFVBQVU7UUFDTyxnQ0FBMkIsR0FBRyxTQUFTLENBQUM7UUFDeEMseUNBQW9DLEdBQUcsRUFBRSxDQUFDO1FBRTNELFdBQVc7UUFDTSx1Q0FBa0MsR0FBRyxTQUFTLENBQUM7UUFDL0MsdUNBQWtDLEdBQUcsV0FBVyxDQUFDO1FBQ2pELDRDQUF1QyxHQUFHLEdBQUcsQ0FBQztRQUc3RCxJQUFJLENBQUMsYUFBYSxHQUFHO1lBQ25CLElBQUksRUFBRSxFQUFFO1lBQ1IsS0FBSyxFQUFFLENBQUM7WUFDUixNQUFNLEVBQUUsQ0FBQztZQUNULFFBQVEsRUFBRSxDQUFDO1lBQ1gscUJBQXFCLEVB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TyxFQUFFLFlBQVk7WUFDckIsTUFBTSxFQUFFLFFBQVE7U0FDakIsQ0FDRixDQUFDO1FBQ0YsSUFBSSxDQUFDLEtBQUssR0FBRyxRQUFRLENBQUM7SUFDeEIsQ0FBQztJQUVPLHNCQUFzQixDQUFDLFVBQWtCO1FBQy9DLFFBQVEsVUFBVSxFQUFFO1lBQ2xCLEtBQUssV0FBVztnQkFDZCxPQUFPLEtBQUssQ0FBQztZQUNmLEtBQUssWUFBWTtnQkFDZixPQUFPLEtBQUssQ0FBQztZQUNmLEtBQUssWUFBWTtnQkFDZixPQUFPLE1BQU0sQ0FBQztZQUNoQjtnQkFDRSxPQUFPLEtBQUssQ0FBQztTQUNoQjtJQUNILENBQUM7O0FBLzZDRCxrQkFBa0I7QUFDSyw0QkFBb0IsR0FBRztJQUM1QyxLQUFLO0lBQ0wsd0JBQXdCO0lBQ3hCLDJCQUEyQjtJQUMzQixLQUFLO0lBQ0wsc0JBQXNCO0lBQ3RCLHlCQUF5QjtDQUMxQixDQUFDO0FBRUYsa0JBQWtCO0FBQ0ssNEJBQW9CLEdBQUc7SUFDNUMsVUFBVTtJQUNWLDJCQUEyQjtJQUMzQixtQ0FBbUM7SUFDbkMsWUFBWTtJQUNaLGlDQUFpQztJQUNqQyx5Q0FBeUM7SUFDekMsVUFBVTtJQUNWLGtDQUFrQztJQUNsQywwQ0FBMEM7SUFDMUMsVUFBVTtJQUNWLGdDQUFnQztJQUNoQyx3Q0FBd0M7SUFDeEMsVUFBVTtJQUNWLGlDQUFpQztJQUNqQyx5Q0FBeUM7SUFDekMsVUFBVTtJQUNWLDBCQUEwQjtJQUMxQixrQ0FBa0M7SUFDbEMsWUFBWTtJQUNaLGdDQUFnQztJQUNoQyx3Q0FBd0M7SUFDeEMsVUFBVTtJQUNWLGlDQUFpQztJQUNqQyx5Q0FBeUM7SUFDekMsVUFBVTtJQUNWLCtCQUErQjtJQUMvQix1Q0FBdUM7SUFDdkMsVUFBVTtJQUNWLGdDQUFnQztJQUNoQyx3Q0FBd0M7Q0FDekMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IFBPU0VfTEFORE1BUktTLCBSZXN1bHRzIH0gZnJvbSAnQG1lZGlhcGlwZS9ob2xpc3RpYyc7XG5pbXBvcnQgKiBhcyBKU1ppcCBmcm9tICdqc3ppcCc7XG5pbXBvcnQgeyBQb3NlU2V0SXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1zZXQtaXRlbSc7XG5pbXBvcnQgeyBQb3NlU2V0SnNvbiB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1zZXQtanNvbic7XG5pbXBvcnQgeyBQb3NlU2V0SnNvbkl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWpzb24taXRlbSc7XG5pbXBvcnQgeyBCb2R5VmVjdG9yIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9ib2R5LXZlY3Rvcic7XG5cbi8vIEB0cy1pZ25vcmVcbmltcG9ydCBjb3NTaW1pbGFyaXR5IGZyb20gJ2Nvcy1zaW1pbGFyaXR5JztcbmltcG9ydCB7IFNpbWlsYXJQb3NlSXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvc2ltaWxhci1wb3NlLWl0ZW0nO1xuaW1wb3J0IHsgSW1hZ2VUcmltbWVyIH0gZnJvbSAnLi9pbnRlcm5hbHMvaW1hZ2UtdHJpbW1lcic7XG5pbXBvcnQgeyBIYW5kVmVjdG9yIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9oYW5kLXZlY3Rvcic7XG5cbmV4cG9ydCBjbGFzcyBQb3NlU2V0IHtcbiAgcHVibGljIGdlbmVyYXRvcj86IHN0cmluZztcbiAgcHVibGljIHZlcnNpb24/OiBudW1iZXI7XG4gIHByaXZhdGUgdmlkZW9NZXRhZGF0YSE6IHtcbiAgICBuYW1lOiBzdHJpbmc7XG4gICAgd2lkdGg6IG51bWJlcjtcbiAgICBoZWlnaHQ6IG51bWJlcjtcbiAgICBkdXJhdGlvbjogbnVtYmVyO1xuICAgIGZpcnN0UG9zZURldGVjdGVkVGltZTogbnVtYmVyO1xuICB9O1xuICBwdWJsaWMgcG9zZXM6IFBvc2VTZXRJdGVtW10gPSBbXTtcbiAgcHVibGljIGlzRmluYWxpemVkPzogYm9vbGVhbiA9IGZhbHNlO1xuXG4gIC8vIEJvZHlWZWN0b3Ig44Gu44Kt44O85ZCNXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgQk9EWV9WRUNUT1JfTUFQUElOR1MgPSBbXG4gICAgLy8g5Y+z6IWVXG4gICAgJ3JpZ2h0V3Jpc3RUb1JpZ2h0RWxib3cnLFxuICAgICdyaWdodEVsYm93VG9SaWdodFNob3VsZGVyJyxcbiAgICAvLyDlt6bohZVcbiAgICAnbGVmdFdyaXN0VG9MZWZ0RWxib3cnLFxuICAgICdsZWZ0RWxib3dUb0xlZnRTaG91bGRlcicsXG4gIF07XG5cbiAgLy8gSGFuZFZlY3RvciDjga7jgq3jg7zlkI1cbiAgcHVibGljIHN0YXRpYyByZWFkb25seSBIQU5EX1ZFQ1RPUl9NQVBQSU5HUyA9IFtcbiAgICAvLyDlj7PmiYsgLSDopqrmjIdcbiAgICAncmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ3JpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5Y+z5omLIC0g5Lq65beu44GX5oyHXG4gICAgJ3JpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOS4reaMh1xuICAgICdyaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ3JpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOiWrOaMh1xuICAgICdyaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5Y+z5omLIC0g5bCP5oyHXG4gICAgJ3JpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOimquaMh1xuICAgICdsZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g5Lq65beu44GX5oyHXG4gICAgJ2xlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlt6bmiYsgLSDkuK3mjIdcbiAgICAnbGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g6Jas5oyHXG4gICAgJ2xlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAnbGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g5bCP5oyHXG4gICAgJ2xlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgXTtcblxuICAvLyDjg53jg7zjgrrjgpLov73liqDjgZnjgovjgZ/jgoHjga7jgq3jg6Xjg7xcbiAgcHJpdmF0ZSBzaW1pbGFyUG9zZVF1ZXVlOiBQb3NlU2V0SXRlbVtdID0gW107XG5cbiAgLy8g6aGe5Ly844Od44O844K644Gu6Zmk5Y67IC0g5YWo44Od44O844K644GL44KJXG4gIHByaXZhdGUgcmVhZG9ubHkgSVNfRU5BQkxFRF9SRU1PVkVfRFVQTElDQVRFRF9QT1NFU19GT1JfV0hPTEUgPSBmYWxzZTtcbiAgLy8g6aGe5Ly844Od44O844K644Gu6Zmk5Y67IC0g5ZCE44Od44O844K644Gu5YmN5b6M44GL44KJXG4gIHByaXZhdGUgcmVhZG9ubHkgSVNfRU5BQkxFRF9SRU1PVkVfRFVQTElDQVRFRF9QT1NFU19GT1JfQVJPVU5EID0gdHJ1ZTtcblxuICAvLyDnlLvlg4/mm7jjgY3lh7rjgZfmmYLjga7oqK3lrppcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9XSURUSDogbnVtYmVyID0gMTA4MDtcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9NSU1FOiAnaW1hZ2UvanBlZycgfCAnaW1hZ2UvcG5nJyB8ICdpbWFnZS93ZWJwJyA9XG4gICAgJ2ltYWdlL3dlYnAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX1FVQUxJVFkgPSAwLjg7XG5cbiAgLy8g55S75YOP44Gu5L2Z55m96Zmk5Y67XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0NPTE9SID0gJyMwMDAwMDAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX01BUkdJTl9UUklNTUlOR19ESUZGX1RIUkVTSE9MRCA9IDUwO1xuXG4gIC8vIOeUu+WDj+OBruiDjOaZr+iJsue9ruaPm1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9TUkNfQ09MT1IgPSAnIzAxNkFGRCc7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RTVF9DT0xPUiA9ICcjRkZGRkZGMDAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9ESUZGX1RIUkVTSE9MRCA9IDEzMDtcblxuICBjb25zdHJ1Y3RvcigpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSB7XG4gICAgICBuYW1lOiAnJyxcbiAgICAgIHdpZHRoOiAwLFxuICAgICAgaGVpZ2h0OiAwLFxuICAgICAgZHVyYXRpb246IDAsXG4gICAgICBmaXJzdFBvc2VEZXRlY3RlZFRpbWU6IDAsXG4gICAgfTtcbiAgfVxuXG4gIGdldFZpZGVvTmFtZSgpIHtcbiAgICByZXR1cm4gdGhpcy52aWRlb01ldGFkYXRhLm5hbWU7XG4gIH1cblxuICBzZXRWaWRlb05hbWUodmlkZW9OYW1lOiBzdHJpbmcpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEubmFtZSA9IHZpZGVvTmFtZTtcbiAgfVxuXG4gIHNldFZpZGVvTWV0YURhdGEod2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIsIGR1cmF0aW9uOiBudW1iZXIpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEud2lkdGggPSB3aWR0aDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuaGVpZ2h0ID0gaGVpZ2h0O1xuICAgIHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbiA9IGR1cmF0aW9uO1xuICB9XG5cbiAgLyoqXG4gICAqIOODneODvOOCuuaVsOOBruWPluW+l1xuICAgKiBAcmV0dXJuc1xuICAgKi9cbiAgZ2V0TnVtYmVyT2ZQb3NlcygpOiBudW1iZXIge1xuICAgIGlmICh0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpIHJldHVybiAtMTtcbiAgICByZXR1cm4gdGhpcy5wb3Nlcy5sZW5ndGg7XG4gIH1cblxuICAvKipcbiAgICog5YWo44Od44O844K644Gu5Y+W5b6XXG4gICAqIEByZXR1cm5zIOWFqOOBpuOBruODneODvOOCulxuICAgKi9cbiAgZ2V0UG9zZXMoKTogUG9zZVNldEl0ZW1bXSB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIFtdO1xuICAgIHJldHVybiB0aGlzLnBvc2VzO1xuICB9XG5cbiAgLyoqXG4gICAqIOaMh+WumuOBleOCjOOBn+aZgumWk+OBq+OCiOOCi+ODneODvOOCuuOBruWPluW+l1xuICAgKiBAcGFyYW0gdGltZU1pbGlzZWNvbmRzIOODneODvOOCuuOBruaZgumWkyAo44Of44Oq56eSKVxuICAgKiBAcmV0dXJucyDjg53jg7zjgrpcbiAgICovXG4gIGdldFBvc2VCeVRpbWUodGltZU1pbGlzZWNvbmRzOiBudW1iZXIpOiBQb3NlU2V0SXRlbSB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICByZXR1cm4gdGhpcy5wb3Nlcy5maW5kKChwb3NlKSA9PiBwb3NlLnRpbWVNaWxpc2Vjb25kcyA9PT0gdGltZU1pbGlzZWNvbmRzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiDjg53jg7zjgrrjga7ov73liqBcbiAgICovXG4gIHB1c2hQb3NlKFxuICAgIHZpZGVvVGltZU1pbGlzZWNvbmRzOiBudW1iZXIsXG4gICAgZnJhbWVJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICBwb3NlSW1hZ2VEYXRhVXJsOiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gICAgZmFjZUZyYW1lSW1hZ2VEYXRhVXJsOiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gICAgcmVzdWx0czogUmVzdWx0c1xuICApOiBQb3NlU2V0SXRlbSB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKHJlc3VsdHMucG9zZUxhbmRtYXJrcyA9PT0gdW5kZWZpbmVkKSByZXR1cm47XG5cbiAgICBpZiAodGhpcy5wb3Nlcy5sZW5ndGggPT09IDApIHtcbiAgICAgIHRoaXMudmlkZW9NZXRhZGF0YS5maXJzdFBvc2VEZXRlY3RlZFRpbWUgPSB2aWRlb1RpbWVNaWxpc2Vjb25kcztcbiAgICB9XG5cbiAgICBjb25zdCBwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZTogYW55W10gPSAocmVzdWx0cyBhcyBhbnkpLmVhXG4gICAgICA/IChyZXN1bHRzIGFzIGFueSkuZWFcbiAgICAgIDogW107XG4gICAgaWYgKHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlLmxlbmd0aCA9PT0gMCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBwb3NlIHdpdGggdGhlIHdvcmxkIGNvb3JkaW5hdGVgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IGJvZHlWZWN0b3IgPSBQb3NlU2V0LmdldEJvZHlWZWN0b3IocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUpO1xuICAgIGlmICghYm9keVZlY3Rvcikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBib2R5IHZlY3RvcmAsXG4gICAgICAgIHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGlmIChcbiAgICAgIHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCAmJlxuICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZFxuICAgICkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBoYW5kIGxhbmRtYXJrc2AsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgfSBlbHNlIGlmIChyZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgbGVmdCBoYW5kIGxhbmRtYXJrc2AsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgfSBlbHNlIGlmIChyZXN1bHRzLnJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIHJpZ2h0IGhhbmQgbGFuZG1hcmtzYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICB9XG5cbiAgICBjb25zdCBoYW5kVmVjdG9yID0gUG9zZVNldC5nZXRIYW5kVmVjdG9yKFxuICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyxcbiAgICAgIHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzXG4gICAgKTtcbiAgICBpZiAoIWhhbmRWZWN0b3IpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgaGFuZCB2ZWN0b3JgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2U6IFBvc2VTZXRJdGVtID0ge1xuICAgICAgdGltZU1pbGlzZWNvbmRzOiB2aWRlb1RpbWVNaWxpc2Vjb25kcyxcbiAgICAgIGR1cmF0aW9uTWlsaXNlY29uZHM6IC0xLFxuICAgICAgcG9zZTogcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubWFwKCh3b3JsZENvb3JkaW5hdGVMYW5kbWFyaykgPT4ge1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLngsXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsueSxcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay56LFxuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLnZpc2liaWxpdHksXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIGxlZnRIYW5kOiByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzPy5tYXAoKG5vcm1hbGl6ZWRMYW5kbWFyaykgPT4ge1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay54LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay55LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay56LFxuICAgICAgICBdO1xuICAgICAgfSksXG4gICAgICByaWdodEhhbmQ6IHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzPy5tYXAoKG5vcm1hbGl6ZWRMYW5kbWFyaykgPT4ge1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay54LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay55LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay56LFxuICAgICAgICBdO1xuICAgICAgfSksXG4gICAgICBib2R5VmVjdG9yOiBib2R5VmVjdG9yLFxuICAgICAgaGFuZFZlY3RvcjogaGFuZFZlY3RvcixcbiAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiBmcmFtZUltYWdlRGF0YVVybCxcbiAgICAgIHBvc2VJbWFnZURhdGFVcmw6IHBvc2VJbWFnZURhdGFVcmwsXG4gICAgICBmYWNlRnJhbWVJbWFnZURhdGFVcmw6IGZhY2VGcmFtZUltYWdlRGF0YVVybCxcbiAgICAgIGV4dGVuZGVkRGF0YToge30sXG4gICAgICBkZWJ1Zzoge1xuICAgICAgICBkdXBsaWNhdGVkSXRlbXM6IFtdLFxuICAgICAgfSxcbiAgICAgIG1lcmdlZFRpbWVNaWxpc2Vjb25kczogLTEsXG4gICAgICBtZXJnZWREdXJhdGlvbk1pbGlzZWNvbmRzOiAtMSxcbiAgICB9O1xuXG4gICAgbGV0IGxhc3RQb3NlO1xuICAgIGlmICh0aGlzLnBvc2VzLmxlbmd0aCA9PT0gMCAmJiAxIDw9IHRoaXMuc2ltaWxhclBvc2VRdWV1ZS5sZW5ndGgpIHtcbiAgICAgIC8vIOmhnuS8vOODneODvOOCuuOCreODpeODvOOBi+OCieacgOW+jOOBruODneODvOOCuuOCkuWPluW+l1xuICAgICAgbGFzdFBvc2UgPSB0aGlzLnNpbWlsYXJQb3NlUXVldWVbdGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCAtIDFdO1xuICAgIH0gZWxzZSBpZiAoMSA8PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgLy8g44Od44O844K66YWN5YiX44GL44KJ5pyA5b6M44Gu44Od44O844K644KS5Y+W5b6XXG4gICAgICBsYXN0UG9zZSA9IHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXTtcbiAgICB9XG5cbiAgICBpZiAobGFzdFBvc2UpIHtcbiAgICAgIC8vIOacgOW+jOOBruODneODvOOCuuOBjOOBguOCjOOBsOOAgemhnuS8vOODneODvOOCuuOBi+OBqeOBhuOBi+OCkuavlOi8g1xuICAgICAgY29uc3QgaXNTaW1pbGFyQm9keVBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckJvZHlQb3NlKFxuICAgICAgICBwb3NlLmJvZHlWZWN0b3IsXG4gICAgICAgIGxhc3RQb3NlLmJvZHlWZWN0b3JcbiAgICAgICk7XG5cbiAgICAgIGxldCBpc1NpbWlsYXJIYW5kUG9zZSA9IHRydWU7XG4gICAgICBpZiAobGFzdFBvc2UuaGFuZFZlY3RvciAmJiBwb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgaXNTaW1pbGFySGFuZFBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckhhbmRQb3NlKFxuICAgICAgICAgIHBvc2UuaGFuZFZlY3RvcixcbiAgICAgICAgICBsYXN0UG9zZS5oYW5kVmVjdG9yXG4gICAgICAgICk7XG4gICAgICB9IGVsc2UgaWYgKCFsYXN0UG9zZS5oYW5kVmVjdG9yICYmIHBvc2UuaGFuZFZlY3Rvcikge1xuICAgICAgICBpc1NpbWlsYXJIYW5kUG9zZSA9IGZhbHNlO1xuICAgICAgfVxuXG4gICAgICBpZiAoIWlzU2ltaWxhckJvZHlQb3NlIHx8ICFpc1NpbWlsYXJIYW5kUG9zZSkge1xuICAgICAgICAvLyDouqvkvZPjg7vmiYvjga7jgYTjgZrjgozjgYvjgYzliY3jga7jg53jg7zjgrrjgajpoZ7kvLzjgZfjgabjgYTjgarjgYTjgarjgonjgbDjgIHpoZ7kvLzjg53jg7zjgrrjgq3jg6Xjg7zjgpLlh6bnkIbjgZfjgabjgIHjg53jg7zjgrrphY3liJfjgbjov73liqBcbiAgICAgICAgdGhpcy5wdXNoUG9zZUZyb21TaW1pbGFyUG9zZVF1ZXVlKHBvc2UudGltZU1pbGlzZWNvbmRzKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyDpoZ7kvLzjg53jg7zjgrrjgq3jg6Xjg7zjgbjov73liqBcbiAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWUucHVzaChwb3NlKTtcblxuICAgIHJldHVybiBwb3NlO1xuICB9XG5cbiAgLyoqXG4gICAqIOODneODvOOCuuOBrumFjeWIl+OBi+OCieODneODvOOCuuOBjOaxuuOBvuOBo+OBpuOBhOOCi+eerOmWk+OCkuWPluW+l1xuICAgKiBAcGFyYW0gcG9zZXMg44Od44O844K644Gu6YWN5YiXXG4gICAqIEByZXR1cm5zIOODneODvOOCuuOBjOaxuuOBvuOBo+OBpuOBhOOCi+eerOmWk1xuICAgKi9cbiAgc3RhdGljIGdldFN1aXRhYmxlUG9zZUJ5UG9zZXMocG9zZXM6IFBvc2VTZXRJdGVtW10pOiBQb3NlU2V0SXRlbSB7XG4gICAgaWYgKHBvc2VzLmxlbmd0aCA9PT0gMCkgcmV0dXJuIG51bGw7XG4gICAgaWYgKHBvc2VzLmxlbmd0aCA9PT0gMSkge1xuICAgICAgcmV0dXJuIHBvc2VzWzFdO1xuICAgIH1cblxuICAgIC8vIOWQhOaomeacrOODneODvOOCuuOBlOOBqOOBrumhnuS8vOW6puOCkuWIneacn+WMllxuICAgIGNvbnN0IHNpbWlsYXJpdGllc09mUG9zZXM6IHtcbiAgICAgIFtrZXk6IG51bWJlcl06IHtcbiAgICAgICAgaGFuZFNpbWlsYXJpdHk6IG51bWJlcjtcbiAgICAgICAgYm9keVNpbWlsYXJpdHk6IG51bWJlcjtcbiAgICAgIH1bXTtcbiAgICB9ID0ge307XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBwb3Nlcy5sZW5ndGg7IGkrKykge1xuICAgICAgc2ltaWxhcml0aWVzT2ZQb3Nlc1twb3Nlc1tpXS50aW1lTWlsaXNlY29uZHNdID0gcG9zZXMubWFwKFxuICAgICAgICAocG9zZTogUG9zZVNldEl0ZW0pID0+IHtcbiAgICAgICAgICByZXR1cm4ge1xuICAgICAgICAgICAgaGFuZFNpbWlsYXJpdHk6IDAsXG4gICAgICAgICAgICBib2R5U2ltaWxhcml0eTogMCxcbiAgICAgICAgICB9O1xuICAgICAgICB9XG4gICAgICApO1xuICAgIH1cblxuICAgIC8vIOWQhOaomeacrOODneODvOOCuuOBlOOBqOOBrumhnuS8vOW6puOCkuioiOeul1xuICAgIGZvciAobGV0IHNhbXBsZVBvc2Ugb2YgcG9zZXMpIHtcbiAgICAgIGxldCBoYW5kU2ltaWxhcml0eTogbnVtYmVyO1xuXG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHBvc2VzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIGNvbnN0IHBvc2UgPSBwb3Nlc1tpXTtcbiAgICAgICAgaWYgKHBvc2UuaGFuZFZlY3RvciAmJiBzYW1wbGVQb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgICBoYW5kU2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0SGFuZFNpbWlsYXJpdHkoXG4gICAgICAgICAgICBwb3NlLmhhbmRWZWN0b3IsXG4gICAgICAgICAgICBzYW1wbGVQb3NlLmhhbmRWZWN0b3JcbiAgICAgICAgICApO1xuICAgICAgICB9XG5cbiAgICAgICAgbGV0IGJvZHlTaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRCb2R5UG9zZVNpbWlsYXJpdHkoXG4gICAgICAgICAgcG9zZS5ib2R5VmVjdG9yLFxuICAgICAgICAgIHNhbXBsZVBvc2UuYm9keVZlY3RvclxuICAgICAgICApO1xuXG4gICAgICAgIHNpbWlsYXJpdGllc09mUG9zZXNbc2FtcGxlUG9zZS50aW1lTWlsaXNlY29uZHNdW2ldID0ge1xuICAgICAgICAgIGhhbmRTaW1pbGFyaXR5OiBoYW5kU2ltaWxhcml0eSA/PyAwLFxuICAgICAgICAgIGJvZHlTaW1pbGFyaXR5LFxuICAgICAgICB9O1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIOmhnuS8vOW6puOBrumrmOOBhOODleODrOODvOODoOOBjOWkmuOBi+OBo+OBn+ODneODvOOCuuOCkumBuOaKnlxuICAgIGNvbnN0IHNpbWlsYXJpdGllc09mU2FtcGxlUG9zZXMgPSBwb3Nlcy5tYXAoKHBvc2U6IFBvc2VTZXRJdGVtKSA9PiB7XG4gICAgICByZXR1cm4gc2ltaWxhcml0aWVzT2ZQb3Nlc1twb3NlLnRpbWVNaWxpc2Vjb25kc10ucmVkdWNlKFxuICAgICAgICAoXG4gICAgICAgICAgcHJldjogbnVtYmVyLFxuICAgICAgICAgIGN1cnJlbnQ6IHsgaGFuZFNpbWlsYXJpdHk6IG51bWJlcjsgYm9keVNpbWlsYXJpdHk6IG51bWJlciB9XG4gICAgICAgICkgPT4ge1xuICAgICAgICAgIHJldHVybiBwcmV2ICsgY3VycmVudC5oYW5kU2ltaWxhcml0eSArIGN1cnJlbnQuYm9keVNpbWlsYXJpdHk7XG4gICAgICAgIH0sXG4gICAgICAgIDBcbiAgICAgICk7XG4gICAgfSk7XG4gICAgY29uc3QgbWF4U2ltaWxhcml0eSA9IE1hdGgubWF4KC4uLnNpbWlsYXJpdGllc09mU2FtcGxlUG9zZXMpO1xuICAgIGNvbnN0IG1heFNpbWlsYXJpdHlJbmRleCA9IHNpbWlsYXJpdGllc09mU2FtcGxlUG9zZXMuaW5kZXhPZihtYXhTaW1pbGFyaXR5KTtcbiAgICBjb25zdCBzZWxlY3RlZFBvc2UgPSBwb3Nlc1ttYXhTaW1pbGFyaXR5SW5kZXhdO1xuICAgIGlmICghc2VsZWN0ZWRQb3NlKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gZ2V0U3VpdGFibGVQb3NlQnlQb3Nlc2AsXG4gICAgICAgIHNpbWlsYXJpdGllc09mU2FtcGxlUG9zZXMsXG4gICAgICAgIG1heFNpbWlsYXJpdHksXG4gICAgICAgIG1heFNpbWlsYXJpdHlJbmRleFxuICAgICAgKTtcbiAgICB9XG5cbiAgICBjb25zb2xlLmRlYnVnKGBbUG9zZVNldF0gZ2V0U3VpdGFibGVQb3NlQnlQb3Nlc2AsIHtcbiAgICAgIHNlbGVjdGVkOiBzZWxlY3RlZFBvc2UsXG4gICAgICB1bnNlbGVjdGVkOiBwb3Nlcy5maWx0ZXIoKHBvc2U6IFBvc2VTZXRJdGVtKSA9PiB7XG4gICAgICAgIHJldHVybiBwb3NlLnRpbWVNaWxpc2Vjb25kcyAhPT0gc2VsZWN0ZWRQb3NlLnRpbWVNaWxpc2Vjb25kcztcbiAgICAgIH0pLFxuICAgIH0pO1xuICAgIHJldHVybiBzZWxlY3RlZFBvc2U7XG4gIH1cblxuICAvKipcbiAgICog5pyA57WC5Yem55CGXG4gICAqICjph43opIfjgZfjgZ/jg53jg7zjgrrjga7pmaTljrvjgIHnlLvlg4/jga7jg57jg7zjgrjjg7PpmaTljrvjgarjgakpXG4gICAqL1xuICBhc3luYyBmaW5hbGl6ZSgpIHtcbiAgICBpZiAodGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCA+IDApIHtcbiAgICAgIC8vIOmhnuS8vOODneODvOOCuuOCreODpeODvOOBq+ODneODvOOCuuOBjOaui+OBo+OBpuOBhOOCi+WgtOWQiOOAgeacgOmBqeOBquODneODvOOCuuOCkumBuOaKnuOBl+OBpuODneODvOOCuumFjeWIl+OBuOi/veWKoFxuICAgICAgdGhpcy5wdXNoUG9zZUZyb21TaW1pbGFyUG9zZVF1ZXVlKHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbik7XG4gICAgfVxuXG4gICAgaWYgKDAgPT0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIC8vIOODneODvOOCuuOBjOS4gOOBpOOCguOBquOBhOWgtOWQiOOAgeWHpueQhuOCkue1guS6hlxuICAgICAgdGhpcy5pc0ZpbmFsaXplZCA9IHRydWU7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgLy8g44Od44O844K644Gu5oyB57aa5pmC6ZaT44KS6Kit5a6aXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLnBvc2VzLmxlbmd0aCAtIDE7IGkrKykge1xuICAgICAgaWYgKHRoaXMucG9zZXNbaV0uZHVyYXRpb25NaWxpc2Vjb25kcyAhPT0gLTEpIGNvbnRpbnVlO1xuICAgICAgdGhpcy5wb3Nlc1tpXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgdGhpcy5wb3Nlc1tpICsgMV0udGltZU1pbGlzZWNvbmRzIC0gdGhpcy5wb3Nlc1tpXS50aW1lTWlsaXNlY29uZHM7XG4gICAgfVxuICAgIHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgIHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbiAtXG4gICAgICB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV0udGltZU1pbGlzZWNvbmRzO1xuXG4gICAgLy8g5YWo5L2T44GL44KJ6YeN6KSH44Od44O844K644KS6Zmk5Y67XG4gICAgaWYgKHRoaXMuSVNfRU5BQkxFRF9SRU1PVkVfRFVQTElDQVRFRF9QT1NFU19GT1JfV0hPTEUpIHtcbiAgICAgIHRoaXMucmVtb3ZlRHVwbGljYXRlZFBvc2VzKCk7XG4gICAgfVxuXG4gICAgLy8g5pyA5Yid44Gu44Od44O844K644KS6Zmk5Y67XG4gICAgdGhpcy5wb3Nlcy5zaGlmdCgpO1xuXG4gICAgLy8g55S75YOP44Gu44Oe44O844K444Oz44KS5Y+W5b6XXG4gICAgY29uc29sZS5kZWJ1ZyhgW1Bvc2VTZXRdIGZpbmFsaXplIC0gRGV0ZWN0aW5nIGltYWdlIG1hcmdpbnMuLi5gKTtcbiAgICBsZXQgaW1hZ2VUcmltbWluZzpcbiAgICAgIHwge1xuICAgICAgICAgIG1hcmdpblRvcDogbnVtYmVyO1xuICAgICAgICAgIG1hcmdpbkJvdHRvbTogbnVtYmVyO1xuICAgICAgICAgIGhlaWdodE5ldzogbnVtYmVyO1xuICAgICAgICAgIGhlaWdodE9sZDogbnVtYmVyO1xuICAgICAgICAgIHdpZHRoOiBudW1iZXI7XG4gICAgICAgIH1cbiAgICAgIHwgdW5kZWZpbmVkID0gdW5kZWZpbmVkO1xuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5mcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGNvbnN0IG1hcmdpbkNvbG9yID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldE1hcmdpbkNvbG9yKCk7XG4gICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gRGV0ZWN0ZWQgbWFyZ2luIGNvbG9yLi4uYCxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgIG1hcmdpbkNvbG9yXG4gICAgICApO1xuICAgICAgaWYgKG1hcmdpbkNvbG9yID09PSBudWxsKSBjb250aW51ZTtcbiAgICAgIGlmIChtYXJnaW5Db2xvciAhPT0gdGhpcy5JTUFHRV9NQVJHSU5fVFJJTU1JTkdfQ09MT1IpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBjb25zdCB0cmltbWVkID0gYXdhaXQgaW1hZ2VUcmltbWVyLnRyaW1NYXJnaW4oXG4gICAgICAgIG1hcmdpbkNvbG9yLFxuICAgICAgICB0aGlzLklNQUdFX01BUkdJTl9UUklNTUlOR19ESUZGX1RIUkVTSE9MRFxuICAgICAgKTtcbiAgICAgIGlmICghdHJpbW1lZCkgY29udGludWU7XG4gICAgICBpbWFnZVRyaW1taW5nID0gdHJpbW1lZDtcbiAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBEZXRlcm1pbmVkIGltYWdlIHRyaW1taW5nIHBvc2l0aW9ucy4uLmAsXG4gICAgICAgIHRyaW1tZWRcbiAgICAgICk7XG4gICAgICBicmVhaztcbiAgICB9XG5cbiAgICAvLyDnlLvlg4/jgpLmlbTlvaJcbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgbGV0IGltYWdlVHJpbW1lciA9IG5ldyBJbWFnZVRyaW1tZXIoKTtcbiAgICAgIGlmICghcG9zZS5mcmFtZUltYWdlRGF0YVVybCB8fCAhcG9zZS5wb3NlSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuXG4gICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gUHJvY2Vzc2luZyBpbWFnZS4uLmAsXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzXG4gICAgICApO1xuXG4gICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDjg5Xjg6zjg7zjg6DnlLvlg49cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UuZnJhbWVJbWFnZURhdGFVcmwpO1xuXG4gICAgICBpZiAoaW1hZ2VUcmltbWluZykge1xuICAgICAgICBhd2FpdCBpbWFnZVRyaW1tZXIuY3JvcChcbiAgICAgICAgICAwLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcubWFyZ2luVG9wLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcud2lkdGgsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5oZWlnaHROZXdcbiAgICAgICAgKTtcbiAgICAgIH1cblxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLnJlcGxhY2VDb2xvcihcbiAgICAgICAgdGhpcy5JTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfU1JDX0NPTE9SLFxuICAgICAgICB0aGlzLklNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9EU1RfQ09MT1IsXG4gICAgICAgIHRoaXMuSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RJRkZfVEhSRVNIT0xEXG4gICAgICApO1xuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVzaXplV2l0aEZpdCh7XG4gICAgICAgIHdpZHRoOiB0aGlzLklNQUdFX1dJRFRILFxuICAgICAgfSk7XG5cbiAgICAgIGxldCBuZXdEYXRhVXJsID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldERhdGFVcmwoXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSxcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2UvanBlZycgfHwgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2Uvd2VicCdcbiAgICAgICAgICA/IHRoaXMuSU1BR0VfUVVBTElUWVxuICAgICAgICAgIDogdW5kZWZpbmVkXG4gICAgICApO1xuICAgICAgaWYgKCFuZXdEYXRhVXJsKSB7XG4gICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gQ291bGQgbm90IGdldCB0aGUgbmV3IGRhdGF1cmwgZm9yIGZyYW1lIGltYWdlYFxuICAgICAgICApO1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIHBvc2UuZnJhbWVJbWFnZURhdGFVcmwgPSBuZXdEYXRhVXJsO1xuXG4gICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDjg53jg7zjgrrjg5fjg6zjg5Pjg6Xjg7znlLvlg49cbiAgICAgIGltYWdlVHJpbW1lciA9IG5ldyBJbWFnZVRyaW1tZXIoKTtcbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UucG9zZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGlmIChpbWFnZVRyaW1taW5nKSB7XG4gICAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5jcm9wKFxuICAgICAgICAgIDAsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5tYXJnaW5Ub3AsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy53aWR0aCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLmhlaWdodE5ld1xuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVzaXplV2l0aEZpdCh7XG4gICAgICAgIHdpZHRoOiB0aGlzLklNQUdFX1dJRFRILFxuICAgICAgfSk7XG5cbiAgICAgIG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgID8gdGhpcy5JTUFHRV9RVUFMSVRZXG4gICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICk7XG4gICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgcG9zZSBwcmV2aWV3IGltYWdlYFxuICAgICAgICApO1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG5cbiAgICAgIGlmIChwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDpoZTjg5Xjg6zjg7zjg6DnlLvlg49cbiAgICAgICAgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgICAgbmV3RGF0YVVybCA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXREYXRhVXJsKFxuICAgICAgICAgIHRoaXMuSU1BR0VfTUlNRSxcbiAgICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgICAgPyB0aGlzLklNQUdFX1FVQUxJVFlcbiAgICAgICAgICAgIDogdW5kZWZpbmVkXG4gICAgICAgICk7XG4gICAgICAgIGlmICghbmV3RGF0YVVybCkge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgZmFjZSBmcmFtZSBpbWFnZWBcbiAgICAgICAgICApO1xuICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICB9XG4gICAgICAgIHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsID0gbmV3RGF0YVVybDtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB0aGlzLmlzRmluYWxpemVkID0gdHJ1ZTtcbiAgfVxuXG4gIC8qKlxuICAgKiDpoZ7kvLzjg53jg7zjgrrjga7lj5blvpdcbiAgICogQHBhcmFtIHJlc3VsdHMgTWVkaWFQaXBlIEhvbGlzdGljIOOBq+OCiOOCi+ODneODvOOCuuOBruaknOWHuue1kOaenFxuICAgKiBAcGFyYW0gdGhyZXNob2xkIOOBl+OBjeOBhOWApFxuICAgKiBAcGFyYW0gdGFyZ2V0UmFuZ2Ug44Od44O844K644KS5q+U6LyD44GZ44KL56+E5ZuyIChhbGw6IOWFqOOBpiwgYm9keVBvc2U6IOi6q+S9k+OBruOBvywgaGFuZFBvc2U6IOaJi+aMh+OBruOBvylcbiAgICogQHJldHVybnMg6aGe5Ly844Od44O844K644Gu6YWN5YiXXG4gICAqL1xuICBnZXRTaW1pbGFyUG9zZXMoXG4gICAgcmVzdWx0czogUmVzdWx0cyxcbiAgICB0aHJlc2hvbGQ6IG51bWJlciA9IDAuOSxcbiAgICB0YXJnZXRSYW5nZTogJ2FsbCcgfCAnYm9keVBvc2UnIHwgJ2hhbmRQb3NlJyA9ICdhbGwnXG4gICk6IFNpbWlsYXJQb3NlSXRlbVtdIHtcbiAgICAvLyDouqvkvZPjga7jg5njgq/jg4jjg6vjgpLlj5blvpdcbiAgICBsZXQgYm9keVZlY3RvcjogQm9keVZlY3RvcjtcbiAgICB0cnkge1xuICAgICAgYm9keVZlY3RvciA9IFBvc2VTZXQuZ2V0Qm9keVZlY3RvcigocmVzdWx0cyBhcyBhbnkpLmVhKTtcbiAgICB9IGNhdGNoIChlKSB7XG4gICAgICBjb25zb2xlLmVycm9yKGBbUG9zZVNldF0gZ2V0U2ltaWxhclBvc2VzIC0gRXJyb3Igb2NjdXJyZWRgLCBlLCByZXN1bHRzKTtcbiAgICAgIHJldHVybiBbXTtcbiAgICB9XG4gICAgaWYgKCFib2R5VmVjdG9yKSB7XG4gICAgICB0aHJvdyAnQ291bGQgbm90IGdldCB0aGUgYm9keSB2ZWN0b3InO1xuICAgIH1cblxuICAgIC8vIOaJi+aMh+OBruODmeOCr+ODiOODq+OCkuWPluW+l1xuICAgIGxldCBoYW5kVmVjdG9yOiBIYW5kVmVjdG9yO1xuICAgIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2FsbCcgfHwgdGFyZ2V0UmFuZ2UgPT09ICdoYW5kUG9zZScpIHtcbiAgICAgIGhhbmRWZWN0b3IgPSBQb3NlU2V0LmdldEhhbmRWZWN0b3IoXG4gICAgICAgIHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3MsXG4gICAgICAgIHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzXG4gICAgICApO1xuICAgICAgaWYgKHRhcmdldFJhbmdlID09PSAnaGFuZFBvc2UnICYmICFoYW5kVmVjdG9yKSB7XG4gICAgICAgIHRocm93ICdDb3VsZCBub3QgZ2V0IHRoZSBoYW5kIHZlY3Rvcic7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8g5ZCE44Od44O844K644Go44OZ44Kv44OI44Or44KS5q+U6LyDXG4gICAgY29uc3QgcG9zZXMgPSBbXTtcbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgaWYgKFxuICAgICAgICAodGFyZ2V0UmFuZ2UgPT09ICdhbGwnIHx8IHRhcmdldFJhbmdlID09PSAnYm9keVBvc2UnKSAmJlxuICAgICAgICAhcG9zZS5ib2R5VmVjdG9yXG4gICAgICApIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9IGVsc2UgaWYgKHRhcmdldFJhbmdlID09PSAnaGFuZFBvc2UnICYmICFwb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG5cbiAgICAgIC8qY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgJ1tQb3NlU2V0XSBnZXRTaW1pbGFyUG9zZXMgLSAnLFxuICAgICAgICB0aGlzLmdldFZpZGVvTmFtZSgpLFxuICAgICAgICBwb3NlLnRpbWVNaWxpc2Vjb25kc1xuICAgICAgKTsqL1xuXG4gICAgICAvLyDouqvkvZPjga7jg53jg7zjgrrjga7poZ7kvLzluqbjgpLlj5blvpdcbiAgICAgIGxldCBib2R5U2ltaWxhcml0eTogbnVtYmVyO1xuICAgICAgaWYgKGJvZHlWZWN0b3IgJiYgcG9zZS5ib2R5VmVjdG9yKSB7XG4gICAgICAgIGJvZHlTaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRCb2R5UG9zZVNpbWlsYXJpdHkoXG4gICAgICAgICAgcG9zZS5ib2R5VmVjdG9yLFxuICAgICAgICAgIGJvZHlWZWN0b3JcbiAgICAgICAgKTtcbiAgICAgIH1cblxuICAgICAgLy8g5omL5oyH44Gu44Od44O844K644Gu6aGe5Ly85bqm44KS5Y+W5b6XXG4gICAgICBsZXQgaGFuZFNpbWlsYXJpdHk6IG51bWJlcjtcbiAgICAgIGlmIChoYW5kVmVjdG9yICYmIHBvc2UuaGFuZFZlY3Rvcikge1xuICAgICAgICBoYW5kU2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0SGFuZFNpbWlsYXJpdHkocG9zZS5oYW5kVmVjdG9yLCBoYW5kVmVjdG9yKTtcbiAgICAgIH1cblxuICAgICAgLy8g5Yik5a6aXG4gICAgICBsZXQgc2ltaWxhcml0eTogbnVtYmVyLFxuICAgICAgICBpc1NpbWlsYXIgPSBmYWxzZTtcbiAgICAgIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2FsbCcpIHtcbiAgICAgICAgc2ltaWxhcml0eSA9IE1hdGgubWF4KGJvZHlTaW1pbGFyaXR5ID8/IDAsIGhhbmRTaW1pbGFyaXR5ID8/IDApO1xuICAgICAgICBpZiAodGhyZXNob2xkIDw9IGJvZHlTaW1pbGFyaXR5IHx8IHRocmVzaG9sZCA8PSBoYW5kU2ltaWxhcml0eSkge1xuICAgICAgICAgIGlzU2ltaWxhciA9IHRydWU7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSBpZiAodGFyZ2V0UmFuZ2UgPT09ICdib2R5UG9zZScpIHtcbiAgICAgICAgc2ltaWxhcml0eSA9IGJvZHlTaW1pbGFyaXR5O1xuICAgICAgICBpZiAodGhyZXNob2xkIDw9IGJvZHlTaW1pbGFyaXR5KSB7XG4gICAgICAgICAgaXNTaW1pbGFyID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2hhbmRQb3NlJykge1xuICAgICAgICBzaW1pbGFyaXR5ID0gaGFuZFNpbWlsYXJpdHk7XG4gICAgICAgIGlmICh0aHJlc2hvbGQgPD0gaGFuZFNpbWlsYXJpdHkpIHtcbiAgICAgICAgICBpc1NpbWlsYXIgPSB0cnVlO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmICghaXNTaW1pbGFyKSBjb250aW51ZTtcblxuICAgICAgLy8g57WQ5p6c44G46L+95YqgXG4gICAgICBwb3Nlcy5wdXNoKHtcbiAgICAgICAgLi4ucG9zZSxcbiAgICAgICAgc2ltaWxhcml0eTogc2ltaWxhcml0eSxcbiAgICAgICAgYm9keVBvc2VTaW1pbGFyaXR5OiBib2R5U2ltaWxhcml0eSxcbiAgICAgICAgaGFuZFBvc2VTaW1pbGFyaXR5OiBoYW5kU2ltaWxhcml0eSxcbiAgICAgIH0gYXMgU2ltaWxhclBvc2VJdGVtKTtcbiAgICB9XG5cbiAgICByZXR1cm4gcG9zZXM7XG4gIH1cblxuICAvKipcbiAgICog6Lqr5L2T44Gu5ae/5Yui44KS6KGo44GZ44OZ44Kv44OI44Or44Gu5Y+W5b6XXG4gICAqIEBwYXJhbSBwb3NlTGFuZG1hcmtzIE1lZGlhUGlwZSBIb2xpc3RpYyDjgaflj5blvpfjgafjgY3jgZ/ouqvkvZPjga7jg6/jg7zjg6vjg4nluqfmqJkgKHJhIOmFjeWIlylcbiAgICogQHJldHVybnMg44OZ44Kv44OI44OrXG4gICAqL1xuICBzdGF0aWMgZ2V0Qm9keVZlY3RvcihcbiAgICBwb3NlTGFuZG1hcmtzOiB7IHg6IG51bWJlcjsgeTogbnVtYmVyOyB6OiBudW1iZXIgfVtdXG4gICk6IEJvZHlWZWN0b3IgfCB1bmRlZmluZWQge1xuICAgIHJldHVybiB7XG4gICAgICByaWdodFdyaXN0VG9SaWdodEVsYm93OiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnosXG4gICAgICBdLFxuICAgICAgcmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcjogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS56LFxuICAgICAgXSxcbiAgICAgIGxlZnRXcmlzdFRvTGVmdEVsYm93OiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnosXG4gICAgICBdLFxuICAgICAgbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXI6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueixcbiAgICAgIF0sXG4gICAgfTtcbiAgfVxuXG4gIC8qKlxuICAgKiDmiYvmjIfjga7lp7/li6LjgpLooajjgZnjg5njgq/jg4jjg6vjga7lj5blvpdcbiAgICogQHBhcmFtIGxlZnRIYW5kTGFuZG1hcmtzIE1lZGlhUGlwZSBIb2xpc3RpYyDjgaflj5blvpfjgafjgY3jgZ/lt6bmiYvjga7mraPopo/ljJbluqfmqJlcbiAgICogQHBhcmFtIHJpZ2h0SGFuZExhbmRtYXJrcyBNZWRpYVBpcGUgSG9saXN0aWMg44Gn5Y+W5b6X44Gn44GN44Gf5Y+z5omL44Gu5q2j6KaP5YyW5bqn5qiZXG4gICAqIEByZXR1cm5zIOODmeOCr+ODiOODq1xuICAgKi9cbiAgc3RhdGljIGdldEhhbmRWZWN0b3IoXG4gICAgbGVmdEhhbmRMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W10sXG4gICAgcmlnaHRIYW5kTGFuZG1hcmtzOiB7IHg6IG51bWJlcjsgeTogbnVtYmVyOyB6OiBudW1iZXIgfVtdXG4gICk6IEhhbmRWZWN0b3IgfCB1bmRlZmluZWQge1xuICAgIGlmIChcbiAgICAgIChyaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwKSAmJlxuICAgICAgKGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwKVxuICAgICkge1xuICAgICAgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICB9XG5cbiAgICByZXR1cm4ge1xuICAgICAgLy8g5Y+z5omLIC0g6Kaq5oyHXG4gICAgICByaWdodFRodW1iVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s0XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzNdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s0XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s0XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1szXS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzJdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1szXS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzJdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1szXS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzJdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5Y+z5omLIC0g5Lq65beu44GX5oyHXG4gICAgICByaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s4XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzddLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s4XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzddLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s4XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzddLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzZdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzZdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzZdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5Y+z5omLIC0g5Lit5oyHXG4gICAgICByaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTJdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTFdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMl0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzEyXS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzExXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIHJpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzExXS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzEwXS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTFdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTBdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDolqzmjIdcbiAgICAgIHJpZ2h0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTZdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTVdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNl0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE2XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIHJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNF0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE0XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTVdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTRdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5Y+z5omLIC0g5bCP5oyHXG4gICAgICByaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1syMF0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzIwXS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMjBdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTldLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOF0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE4XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTldLnogLSByaWdodEhhbmRMYW5kbWFya3NbMThdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g6Kaq5oyHXG4gICAgICBsZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzRdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1szXS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s0XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbM10ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbNF0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1syXS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1szXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMl0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbM10ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzJdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g5Lq65beu44GX5oyHXG4gICAgICBsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzhdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1s3XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s4XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbN10ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbOF0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzddLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzddLnggLSBsZWZ0SGFuZExhbmRtYXJrc1s2XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s3XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbNl0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbN10ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzZdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g5Lit5oyHXG4gICAgICBsZWZ0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMl0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzExXS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMl0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzExXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMl0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzExXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIGxlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTFdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxMF0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTFdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxMF0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTFdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxMF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDolqzmjIdcbiAgICAgIGxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNl0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNl0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNl0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIGxlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTRdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTRdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTRdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g5bCP5oyHXG4gICAgICBsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzIwXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTldLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzIwXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTldLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzIwXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTldLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMThdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMThdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMThdLnosXG4gICAgICAgICAgICBdLFxuICAgIH07XG4gIH1cblxuICAvKipcbiAgICogQm9keVZlY3RvciDplpPjgYzpoZ7kvLzjgZfjgabjgYTjgovjgYvjganjgYbjgYvjga7liKTlrppcbiAgICogQHBhcmFtIGJvZHlWZWN0b3JBIOavlOi8g+WFiOOBriBCb2R5VmVjdG9yXG4gICAqIEBwYXJhbSBib2R5VmVjdG9yQiDmr5TovIPlhYPjga4gQm9keVZlY3RvclxuICAgKiBAcGFyYW0gdGhyZXNob2xkIOOBl+OBjeOBhOWApFxuICAgKiBAcmV0dXJucyDpoZ7kvLzjgZfjgabjgYTjgovjgYvjganjgYbjgYtcbiAgICovXG4gIHN0YXRpYyBpc1NpbWlsYXJCb2R5UG9zZShcbiAgICBib2R5VmVjdG9yQTogQm9keVZlY3RvcixcbiAgICBib2R5VmVjdG9yQjogQm9keVZlY3RvcixcbiAgICB0aHJlc2hvbGQgPSAwLjhcbiAgKTogYm9vbGVhbiB7XG4gICAgbGV0IGlzU2ltaWxhciA9IGZhbHNlO1xuICAgIGNvbnN0IHNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEJvZHlQb3NlU2ltaWxhcml0eShib2R5VmVjdG9yQSwgYm9keVZlY3RvckIpO1xuICAgIGlmIChzaW1pbGFyaXR5ID49IHRocmVzaG9sZCkgaXNTaW1pbGFyID0gdHJ1ZTtcblxuICAgIC8vIGNvbnNvbGUuZGVidWcoYFtQb3NlU2V0XSBpc1NpbWlsYXJQb3NlYCwgaXNTaW1pbGFyLCBzaW1pbGFyaXR5KTtcblxuICAgIHJldHVybiBpc1NpbWlsYXI7XG4gIH1cblxuICAvKipcbiAgICog6Lqr5L2T44Od44O844K644Gu6aGe5Ly85bqm44Gu5Y+W5b6XXG4gICAqIEBwYXJhbSBib2R5VmVjdG9yQSDmr5TovIPlhYjjga4gQm9keVZlY3RvclxuICAgKiBAcGFyYW0gYm9keVZlY3RvckIg5q+U6LyD5YWD44GuIEJvZHlWZWN0b3JcbiAgICogQHJldHVybnMg6aGe5Ly85bqmXG4gICAqL1xuICBzdGF0aWMgZ2V0Qm9keVBvc2VTaW1pbGFyaXR5KFxuICAgIGJvZHlWZWN0b3JBOiBCb2R5VmVjdG9yLFxuICAgIGJvZHlWZWN0b3JCOiBCb2R5VmVjdG9yXG4gICk6IG51bWJlciB7XG4gICAgY29uc3QgY29zU2ltaWxhcml0aWVzID0ge1xuICAgICAgbGVmdFdyaXN0VG9MZWZ0RWxib3c6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLmxlZnRXcmlzdFRvTGVmdEVsYm93LFxuICAgICAgICBib2R5VmVjdG9yQi5sZWZ0V3Jpc3RUb0xlZnRFbGJvd1xuICAgICAgKSxcbiAgICAgIGxlZnRFbGJvd1RvTGVmdFNob3VsZGVyOiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5sZWZ0RWxib3dUb0xlZnRTaG91bGRlcixcbiAgICAgICAgYm9keVZlY3RvckIubGVmdEVsYm93VG9MZWZ0U2hvdWxkZXJcbiAgICAgICksXG4gICAgICByaWdodFdyaXN0VG9SaWdodEVsYm93OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5yaWdodFdyaXN0VG9SaWdodEVsYm93LFxuICAgICAgICBib2R5VmVjdG9yQi5yaWdodFdyaXN0VG9SaWdodEVsYm93XG4gICAgICApLFxuICAgICAgcmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcjogY29zU2ltaWxhcml0eShcbiAgICAgICAgYm9keVZlY3RvckEucmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcixcbiAgICAgICAgYm9keVZlY3RvckIucmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlclxuICAgICAgKSxcbiAgICB9O1xuXG4gICAgY29uc3QgY29zU2ltaWxhcml0aWVzU3VtID0gT2JqZWN0LnZhbHVlcyhjb3NTaW1pbGFyaXRpZXMpLnJlZHVjZShcbiAgICAgIChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSxcbiAgICAgIDBcbiAgICApO1xuICAgIHJldHVybiBjb3NTaW1pbGFyaXRpZXNTdW0gLyBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXMpLmxlbmd0aDtcbiAgfVxuXG4gIC8qKlxuICAgKiBIYW5kVmVjdG9yIOmWk+OBjOmhnuS8vOOBl+OBpuOBhOOCi+OBi+OBqeOBhuOBi+OBruWIpOWumlxuICAgKiBAcGFyYW0gaGFuZFZlY3RvckEg5q+U6LyD5YWI44GuIEhhbmRWZWN0b3JcbiAgICogQHBhcmFtIGhhbmRWZWN0b3JCIOavlOi8g+WFg+OBriBIYW5kVmVjdG9yXG4gICAqIEBwYXJhbSB0aHJlc2hvbGQg44GX44GN44GE5YCkXG4gICAqIEByZXR1cm5zIOmhnuS8vOOBl+OBpuOBhOOCi+OBi+OBqeOBhuOBi1xuICAgKi9cbiAgc3RhdGljIGlzU2ltaWxhckhhbmRQb3NlKFxuICAgIGhhbmRWZWN0b3JBOiBIYW5kVmVjdG9yLFxuICAgIGhhbmRWZWN0b3JCOiBIYW5kVmVjdG9yLFxuICAgIHRocmVzaG9sZCA9IDAuNzVcbiAgKTogYm9vbGVhbiB7XG4gICAgY29uc3Qgc2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0SGFuZFNpbWlsYXJpdHkoaGFuZFZlY3RvckEsIGhhbmRWZWN0b3JCKTtcbiAgICBpZiAoc2ltaWxhcml0eSA9PT0gLTEpIHtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cbiAgICByZXR1cm4gc2ltaWxhcml0eSA+PSB0aHJlc2hvbGQ7XG4gIH1cblxuICAvKipcbiAgICog5omL44Gu44Od44O844K644Gu6aGe5Ly85bqm44Gu5Y+W5b6XXG4gICAqIEBwYXJhbSBoYW5kVmVjdG9yQSDmr5TovIPlhYjjga4gSGFuZFZlY3RvclxuICAgKiBAcGFyYW0gaGFuZFZlY3RvckIg5q+U6LyD5YWD44GuIEhhbmRWZWN0b3JcbiAgICogQHJldHVybnMg6aGe5Ly85bqmXG4gICAqL1xuICBzdGF0aWMgZ2V0SGFuZFNpbWlsYXJpdHkoXG4gICAgaGFuZFZlY3RvckE6IEhhbmRWZWN0b3IsXG4gICAgaGFuZFZlY3RvckI6IEhhbmRWZWN0b3JcbiAgKTogbnVtYmVyIHtcbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQgPVxuICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ID09PSBudWxsIHx8XG4gICAgICBoYW5kVmVjdG9yQi5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICAgPyB1bmRlZmluZWRcbiAgICAgICAgOiB7XG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDopqrmjIdcbiAgICAgICAgICAgIHJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5Y+z5omLIC0g5Lq65beu44GX5oyHXG4gICAgICAgICAgICByaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOS4reaMh1xuICAgICAgICAgICAgcmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDolqzmjIdcbiAgICAgICAgICAgIHJpZ2h0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDlsI/mjIdcbiAgICAgICAgICAgIHJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgIH07XG5cbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCA9XG4gICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbCB8fFxuICAgICAgaGFuZFZlY3RvckIubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICAgPyB1bmRlZmluZWRcbiAgICAgICAgOiB7XG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAgICAgICAgIGxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5Lq65beu44GX5oyHXG4gICAgICAgICAgICBsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOS4reaMh1xuICAgICAgICAgICAgbGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIGxlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDolqzmjIdcbiAgICAgICAgICAgIGxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5bCP5oyHXG4gICAgICAgICAgICBsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICB9O1xuXG4gICAgLy8g5bem5omL44Gu6aGe5Ly85bqmXG4gICAgbGV0IGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kID0gMDtcbiAgICBpZiAoY29zU2ltaWxhcml0aWVzTGVmdEhhbmQpIHtcbiAgICAgIGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kID0gT2JqZWN0LnZhbHVlcyhcbiAgICAgICAgY29zU2ltaWxhcml0aWVzTGVmdEhhbmRcbiAgICAgICkucmVkdWNlKChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSwgMCk7XG4gICAgfVxuXG4gICAgLy8g5Y+z5omL44Gu6aGe5Ly85bqmXG4gICAgbGV0IGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCA9IDA7XG4gICAgaWYgKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCkge1xuICAgICAgY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kID0gT2JqZWN0LnZhbHVlcyhcbiAgICAgICAgY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kXG4gICAgICApLnJlZHVjZSgoc3VtLCB2YWx1ZSkgPT4gc3VtICsgdmFsdWUsIDApO1xuICAgIH1cblxuICAgIC8vIOWQiOeul+OBleOCjOOBn+mhnuS8vOW6plxuICAgIGlmIChjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQgJiYgY29zU2ltaWxhcml0aWVzTGVmdEhhbmQpIHtcbiAgICAgIHJldHVybiAoXG4gICAgICAgIChjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgKyBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCkgL1xuICAgICAgICAoT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kISkubGVuZ3RoICtcbiAgICAgICAgICBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCEpLmxlbmd0aClcbiAgICAgICk7XG4gICAgfSBlbHNlIGlmIChjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQpIHtcbiAgICAgIGlmIChcbiAgICAgICAgaGFuZFZlY3RvckIubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgIT09IG51bGwgJiZcbiAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICkge1xuICAgICAgICAvLyBoYW5kVmVjdG9yQiDjgaflt6bmiYvjgYzjgYLjgovjga7jgasgaGFuZFZlY3RvckEg44Gn5bem5omL44GM44Gq44GE5aC05ZCI44CB6aGe5Ly85bqm44KS5rib44KJ44GZXG4gICAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgICAgYFtQb3NlU2V0XSBnZXRIYW5kU2ltaWxhcml0eSAtIEFkanVzdCBzaW1pbGFyaXR5LCBiZWNhdXNlIGxlZnQgaGFuZCBub3QgZm91bmQuLi5gXG4gICAgICAgICk7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kIC9cbiAgICAgICAgICAoT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kISkubGVuZ3RoICogMilcbiAgICAgICAgKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiAoXG4gICAgICAgIGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCAvXG4gICAgICAgIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCEpLmxlbmd0aFxuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kKSB7XG4gICAgICBpZiAoXG4gICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCAhPT0gbnVsbCAmJlxuICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICkge1xuICAgICAgICAvLyBoYW5kVmVjdG9yQiDjgaflj7PmiYvjgYzjgYLjgovjga7jgasgaGFuZFZlY3RvckEg44Gn5Y+z5omL44GM44Gq44GE5aC05ZCI44CB6aGe5Ly85bqm44KS5rib44KJ44GZXG4gICAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgICAgYFtQb3NlU2V0XSBnZXRIYW5kU2ltaWxhcml0eSAtIEFkanVzdCBzaW1pbGFyaXR5LCBiZWNhdXNlIHJpZ2h0IGhhbmQgbm90IGZvdW5kLi4uYFxuICAgICAgICApO1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgIGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kIC9cbiAgICAgICAgICAoT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzTGVmdEhhbmQhKS5sZW5ndGggKiAyKVxuICAgICAgICApO1xuICAgICAgfVxuICAgICAgcmV0dXJuIChcbiAgICAgICAgY29zU2ltaWxhcml0aWVzU3VtTGVmdEhhbmQgL1xuICAgICAgICBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCEpLmxlbmd0aFxuICAgICAgKTtcbiAgICB9XG5cbiAgICByZXR1cm4gLTE7XG4gIH1cblxuICAvKipcbiAgICogWklQIOODleOCoeOCpOODq+OBqOOBl+OBpuOBruOCt+ODquOCouODqeOCpOOCulxuICAgKiBAcmV0dXJucyBaSVDjg5XjgqHjgqTjg6sgKEJsb2Ig5b2i5byPKVxuICAgKi9cbiAgcHVibGljIGFzeW5jIGdldFppcCgpOiBQcm9taXNlPEJsb2I+IHtcbiAgICBjb25zdCBqc1ppcCA9IG5ldyBKU1ppcCgpO1xuICAgIGpzWmlwLmZpbGUoJ3Bvc2VzLmpzb24nLCBhd2FpdCB0aGlzLmdldEpzb24oKSk7XG5cbiAgICBjb25zdCBpbWFnZUZpbGVFeHQgPSB0aGlzLmdldEZpbGVFeHRlbnNpb25CeU1pbWUodGhpcy5JTUFHRV9NSU1FKTtcblxuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBpZiAocG9zZS5mcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UuZnJhbWVJbWFnZURhdGFVcmwuaW5kZXhPZignYmFzZTY0LCcpICsgJ2Jhc2U2NCwnLmxlbmd0aDtcbiAgICAgICAgICBjb25zdCBiYXNlNjQgPSBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsLnN1YnN0cmluZyhpbmRleCk7XG4gICAgICAgICAganNaaXAuZmlsZShgZnJhbWUtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtpbWFnZUZpbGVFeHR9YCwgYmFzZTY0LCB7XG4gICAgICAgICAgICBiYXNlNjQ6IHRydWUsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlRXhwb3J0ZXJTZXJ2aWNlXSBwdXNoIC0gQ291bGQgbm90IHB1c2ggZnJhbWUgaW1hZ2VgLFxuICAgICAgICAgICAgZXJyb3JcbiAgICAgICAgICApO1xuICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBpZiAocG9zZS5wb3NlSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgY29uc3QgaW5kZXggPVxuICAgICAgICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5wb3NlSW1hZ2VEYXRhVXJsLnN1YnN0cmluZyhpbmRleCk7XG4gICAgICAgICAganNaaXAuZmlsZShgcG9zZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ltYWdlRmlsZUV4dH1gLCBiYXNlNjQsIHtcbiAgICAgICAgICAgIGJhc2U2NDogdHJ1ZSxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgW1Bvc2VFeHBvcnRlclNlcnZpY2VdIHB1c2ggLSBDb3VsZCBub3QgcHVzaCBmcmFtZSBpbWFnZWAsXG4gICAgICAgICAgICBlcnJvclxuICAgICAgICAgICk7XG4gICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGlmIChwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcbiAgICAgICAgICBqc1ppcC5maWxlKGBmYWNlLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7aW1hZ2VGaWxlRXh0fWAsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZhY2UgZnJhbWUgaW1hZ2VgLFxuICAgICAgICAgICAgZXJyb3JcbiAgICAgICAgICApO1xuICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGF3YWl0IGpzWmlwLmdlbmVyYXRlQXN5bmMoeyB0eXBlOiAnYmxvYicgfSk7XG4gIH1cblxuICAvKipcbiAgICogSlNPTiDmloflrZfliJfjgajjgZfjgabjga7jgrfjg6rjgqLjg6njgqTjgrpcbiAgICogQHJldHVybnMgSlNPTiDmloflrZfliJdcbiAgICovXG4gIHB1YmxpYyBhc3luYyBnZXRKc29uKCk6IFByb21pc2U8c3RyaW5nPiB7XG4gICAgaWYgKHRoaXMudmlkZW9NZXRhZGF0YSA9PT0gdW5kZWZpbmVkIHx8IHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZClcbiAgICAgIHJldHVybiAne30nO1xuXG4gICAgaWYgKCF0aGlzLmlzRmluYWxpemVkKSB7XG4gICAgICBhd2FpdCB0aGlzLmZpbmFsaXplKCk7XG4gICAgfVxuXG4gICAgbGV0IHBvc2VMYW5kbWFya01hcHBpbmdzID0gW107XG4gICAgZm9yIChjb25zdCBrZXkgb2YgT2JqZWN0LmtleXMoUE9TRV9MQU5ETUFSS1MpKSB7XG4gICAgICBjb25zdCBpbmRleDogbnVtYmVyID0gUE9TRV9MQU5ETUFSS1Nba2V5IGFzIGtleW9mIHR5cGVvZiBQT1NFX0xBTkRNQVJLU107XG4gICAgICBwb3NlTGFuZG1hcmtNYXBwaW5nc1tpbmRleF0gPSBrZXk7XG4gICAgfVxuXG4gICAgY29uc3QganNvbjogUG9zZVNldEpzb24gPSB7XG4gICAgICBnZW5lcmF0b3I6ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicsXG4gICAgICB2ZXJzaW9uOiAxLFxuICAgICAgdmlkZW86IHRoaXMudmlkZW9NZXRhZGF0YSEsXG4gICAgICBwb3NlczogdGhpcy5wb3Nlcy5tYXAoKHBvc2U6IFBvc2VTZXRJdGVtKTogUG9zZVNldEpzb25JdGVtID0+IHtcbiAgICAgICAgLy8gQm9keVZlY3RvciDjga7lnKfnuK5cbiAgICAgICAgY29uc3QgYm9keVZlY3RvciA9IFtdO1xuICAgICAgICBmb3IgKGNvbnN0IGtleSBvZiBQb3NlU2V0LkJPRFlfVkVDVE9SX01BUFBJTkdTKSB7XG4gICAgICAgICAgYm9keVZlY3Rvci5wdXNoKHBvc2UuYm9keVZlY3RvcltrZXkgYXMga2V5b2YgQm9keVZlY3Rvcl0pO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gSGFuZFZlY3RvciDjga7lnKfnuK5cbiAgICAgICAgbGV0IGhhbmRWZWN0b3I6IChudW1iZXJbXSB8IG51bGwpW10gfCB1bmRlZmluZWQgPSB1bmRlZmluZWQ7XG4gICAgICAgIGlmIChwb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgICBoYW5kVmVjdG9yID0gW107XG4gICAgICAgICAgZm9yIChjb25zdCBrZXkgb2YgUG9zZVNldC5IQU5EX1ZFQ1RPUl9NQVBQSU5HUykge1xuICAgICAgICAgICAgaGFuZFZlY3Rvci5wdXNoKHBvc2UuaGFuZFZlY3RvcltrZXkgYXMga2V5b2YgSGFuZFZlY3Rvcl0pO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuXG4gICAgICAgIC8vIFBvc2VTZXRKc29uSXRlbSDjga4gcG9zZSDjgqrjg5bjgrjjgqfjgq/jg4jjgpLnlJ/miJBcbiAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICB0OiBwb3NlLnRpbWVNaWxpc2Vjb25kcyxcbiAgICAgICAgICBkOiBwb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMsXG4gICAgICAgICAgcDogcG9zZS5wb3NlLFxuICAgICAgICAgIGw6IHBvc2UubGVmdEhhbmQsXG4gICAgICAgICAgcjogcG9zZS5yaWdodEhhbmQsXG4gICAgICAgICAgdjogYm9keVZlY3RvcixcbiAgICAgICAgICBoOiBoYW5kVmVjdG9yLFxuICAgICAgICAgIGU6IHBvc2UuZXh0ZW5kZWREYXRhLFxuICAgICAgICAgIG1kOiBwb3NlLm1lcmdlZER1cmF0aW9uTWlsaXNlY29uZHMsXG4gICAgICAgICAgbXQ6IHBvc2UubWVyZ2VkVGltZU1pbGlzZWNvbmRzLFxuICAgICAgICB9O1xuICAgICAgfSksXG4gICAgICBwb3NlTGFuZG1hcmtNYXBwcGluZ3M6IHBvc2VMYW5kbWFya01hcHBpbmdzLFxuICAgIH07XG5cbiAgICByZXR1cm4gSlNPTi5zdHJpbmdpZnkoanNvbik7XG4gIH1cblxuICAvKipcbiAgICogSlNPTiDjgYvjgonjga7oqq3jgb/ovrzjgb9cbiAgICogQHBhcmFtIGpzb24gSlNPTiDmloflrZfliJcg44G+44Gf44GvIEpTT04g44Kq44OW44K444Kn44Kv44OIXG4gICAqL1xuICBsb2FkSnNvbihqc29uOiBzdHJpbmcgfCBhbnkpIHtcbiAgICBjb25zdCBwYXJzZWRKc29uID0gdHlwZW9mIGpzb24gPT09ICdzdHJpbmcnID8gSlNPTi5wYXJzZShqc29uKSA6IGpzb247XG5cbiAgICBpZiAocGFyc2VkSnNvbi5nZW5lcmF0b3IgIT09ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicpIHtcbiAgICAgIHRocm93ICfkuI3mraPjgarjg5XjgqHjgqTjg6snO1xuICAgIH0gZWxzZSBpZiAocGFyc2VkSnNvbi52ZXJzaW9uICE9PSAxKSB7XG4gICAgICB0aHJvdyAn5pyq5a++5b+c44Gu44OQ44O844K444On44OzJztcbiAgICB9XG5cbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSBwYXJzZWRKc29uLnZpZGVvO1xuICAgIHRoaXMucG9zZXMgPSBwYXJzZWRKc29uLnBvc2VzLm1hcCgoaXRlbTogUG9zZVNldEpzb25JdGVtKTogUG9zZVNldEl0ZW0gPT4ge1xuICAgICAgY29uc3QgYm9keVZlY3RvcjogYW55ID0ge307XG4gICAgICBQb3NlU2V0LkJPRFlfVkVDVE9SX01BUFBJTkdTLm1hcCgoa2V5LCBpbmRleCkgPT4ge1xuICAgICAgICBib2R5VmVjdG9yW2tleSBhcyBrZXlvZiBCb2R5VmVjdG9yXSA9IGl0ZW0udltpbmRleF07XG4gICAgICB9KTtcblxuICAgICAgY29uc3QgaGFuZFZlY3RvcjogYW55ID0ge307XG4gICAgICBpZiAoaXRlbS5oKSB7XG4gICAgICAgIFBvc2VTZXQuSEFORF9WRUNUT1JfTUFQUElOR1MubWFwKChrZXksIGluZGV4KSA9PiB7XG4gICAgICAgICAgaGFuZFZlY3RvcltrZXkgYXMga2V5b2YgSGFuZFZlY3Rvcl0gPSBpdGVtLmghW2luZGV4XTtcbiAgICAgICAgfSk7XG4gICAgICB9XG5cbiAgICAgIHJldHVybiB7XG4gICAgICAgIHRpbWVNaWxpc2Vjb25kczogaXRlbS50LFxuICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBpdGVtLmQsXG4gICAgICAgIHBvc2U6IGl0ZW0ucCxcbiAgICAgICAgbGVmdEhhbmQ6IGl0ZW0ubCxcbiAgICAgICAgcmlnaHRIYW5kOiBpdGVtLnIsXG4gICAgICAgIGJvZHlWZWN0b3I6IGJvZHlWZWN0b3IsXG4gICAgICAgIGhhbmRWZWN0b3I6IGhhbmRWZWN0b3IsXG4gICAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiB1bmRlZmluZWQsXG4gICAgICAgIGV4dGVuZGVkRGF0YTogaXRlbS5lLFxuICAgICAgICBkZWJ1ZzogdW5kZWZpbmVkLFxuICAgICAgICBtZXJnZWREdXJhdGlvbk1pbGlzZWNvbmRzOiBpdGVtLm1kLFxuICAgICAgICBtZXJnZWRUaW1lTWlsaXNlY29uZHM6IGl0ZW0ubXQsXG4gICAgICB9O1xuICAgIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIFpJUCDjg5XjgqHjgqTjg6vjgYvjgonjga7oqq3jgb/ovrzjgb9cbiAgICogQHBhcmFtIGJ1ZmZlciBaSVAg44OV44Kh44Kk44Or44GuIEJ1ZmZlclxuICAgKiBAcGFyYW0gaW5jbHVkZUltYWdlcyDnlLvlg4/jgpLlsZXplovjgZnjgovjgYvjganjgYbjgYtcbiAgICovXG4gIGFzeW5jIGxvYWRaaXAoYnVmZmVyOiBBcnJheUJ1ZmZlciwgaW5jbHVkZUltYWdlczogYm9vbGVhbiA9IHRydWUpIHtcbiAgICBjb25zdCBqc1ppcCA9IG5ldyBKU1ppcCgpO1xuICAgIGNvbnNvbGUuZGVidWcoYFtQb3NlU2V0XSBpbml0Li4uYCk7XG4gICAgY29uc3QgemlwID0gYXdhaXQganNaaXAubG9hZEFzeW5jKGJ1ZmZlciwgeyBiYXNlNjQ6IGZhbHNlIH0pO1xuICAgIGlmICghemlwKSB0aHJvdyAnWklQ44OV44Kh44Kk44Or44KS6Kqt44G/6L6844KB44G+44Gb44KT44Gn44GX44GfJztcblxuICAgIGNvbnN0IGpzb24gPSBhd2FpdCB6aXAuZmlsZSgncG9zZXMuanNvbicpPy5hc3luYygndGV4dCcpO1xuICAgIGlmIChqc29uID09PSB1bmRlZmluZWQpIHtcbiAgICAgIHRocm93ICdaSVDjg5XjgqHjgqTjg6vjgasgcG9zZS5qc29uIOOBjOWQq+OBvuOCjOOBpuOBhOOBvuOBm+OCkyc7XG4gICAgfVxuXG4gICAgdGhpcy5sb2FkSnNvbihqc29uKTtcblxuICAgIGNvbnN0IGZpbGVFeHQgPSB0aGlzLmdldEZpbGVFeHRlbnNpb25CeU1pbWUodGhpcy5JTUFHRV9NSU1FKTtcblxuICAgIGlmIChpbmNsdWRlSW1hZ2VzKSB7XG4gICAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgICBjb25zdCBmcmFtZUltYWdlRmlsZU5hbWUgPSBgZnJhbWUtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtmaWxlRXh0fWA7XG4gICAgICAgICAgY29uc3QgaW1hZ2VCYXNlNjQgPSBhd2FpdCB6aXBcbiAgICAgICAgICAgIC5maWxlKGZyYW1lSW1hZ2VGaWxlTmFtZSlcbiAgICAgICAgICAgID8uYXN5bmMoJ2Jhc2U2NCcpO1xuICAgICAgICAgIGlmIChpbWFnZUJhc2U2NCkge1xuICAgICAgICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybCA9IGBkYXRhOiR7dGhpcy5JTUFHRV9NSU1FfTtiYXNlNjQsJHtpbWFnZUJhc2U2NH1gO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBpZiAoIXBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICAgIGNvbnN0IHBvc2VJbWFnZUZpbGVOYW1lID0gYHBvc2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtmaWxlRXh0fWA7XG4gICAgICAgICAgY29uc3QgaW1hZ2VCYXNlNjQgPSBhd2FpdCB6aXBcbiAgICAgICAgICAgIC5maWxlKHBvc2VJbWFnZUZpbGVOYW1lKVxuICAgICAgICAgICAgPy5hc3luYygnYmFzZTY0Jyk7XG4gICAgICAgICAgaWYgKGltYWdlQmFzZTY0KSB7XG4gICAgICAgICAgICBwb3NlLnBvc2VJbWFnZURhdGFVcmwgPSBgZGF0YToke3RoaXMuSU1BR0VfTUlNRX07YmFzZTY0LCR7aW1hZ2VCYXNlNjR9YDtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICBwcml2YXRlIHB1c2hQb3NlRnJvbVNpbWlsYXJQb3NlUXVldWUobmV4dFBvc2VUaW1lTWlsaXNlY29uZHM/OiBudW1iZXIpIHtcbiAgICBpZiAodGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCA9PT0gMCkgcmV0dXJuO1xuXG4gICAgaWYgKHRoaXMuc2ltaWxhclBvc2VRdWV1ZS5sZW5ndGggPT09IDEpIHtcbiAgICAgIC8vIOmhnuS8vOODneODvOOCuuOCreODpeODvOOBq+ODneODvOOCuuOBjOS4gOOBpOOBl+OBi+OBquOBhOWgtOWQiOOAgeW9k+ipsuODneODvOOCuuOCkuODneODvOOCuumFjeWIl+OBuOi/veWKoFxuICAgICAgY29uc3QgcG9zZSA9IHRoaXMuc2ltaWxhclBvc2VRdWV1ZVswXTtcbiAgICAgIHRoaXMucG9zZXMucHVzaChwb3NlKTtcbiAgICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZSA9IFtdO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIC8vIOWQhOODneODvOOCuuOBruaMgee2muaZgumWk+OCkuioreWumlxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCAtIDE7IGkrKykge1xuICAgICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlW2ldLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWVbaSArIDFdLnRpbWVNaWxpc2Vjb25kcyAtXG4gICAgICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZVtpXS50aW1lTWlsaXNlY29uZHM7XG4gICAgfVxuICAgIGlmIChuZXh0UG9zZVRpbWVNaWxpc2Vjb25kcykge1xuICAgICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlW1xuICAgICAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWUubGVuZ3RoIC0gMVxuICAgICAgXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgbmV4dFBvc2VUaW1lTWlsaXNlY29uZHMgLVxuICAgICAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWVbdGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCAtIDFdLnRpbWVNaWxpc2Vjb25kcztcbiAgICB9XG5cbiAgICAvLyDpoZ7kvLzjg53jg7zjgrrjgq3jg6Xjg7zjga7kuK3jgYvjgonmnIDjgoLmjIHntprmmYLplpPjgYzplbfjgYTjg53jg7zjgrrjgpLpgbjmip5cbiAgICBjb25zdCBzZWxlY3RlZFBvc2UgPSBQb3NlU2V0LmdldFN1aXRhYmxlUG9zZUJ5UG9zZXModGhpcy5zaW1pbGFyUG9zZVF1ZXVlKTtcblxuICAgIC8vIOmBuOaKnuOBleOCjOOBquOBi+OBo+OBn+ODneODvOOCuuOCkuWIl+aMmVxuICAgIHNlbGVjdGVkUG9zZS5kZWJ1Zy5kdXBsaWNhdGVkSXRlbXMgPSB0aGlzLnNpbWlsYXJQb3NlUXVldWVcbiAgICAgIC5maWx0ZXIoKGl0ZW06IFBvc2VTZXRJdGVtKSA9PiB7XG4gICAgICAgIHJldHVybiBpdGVtLnRpbWVNaWxpc2Vjb25kcyAhPT0gc2VsZWN0ZWRQb3NlLnRpbWVNaWxpc2Vjb25kcztcbiAgICAgIH0pXG4gICAgICAubWFwKChpdGVtOiBQb3NlU2V0SXRlbSkgPT4ge1xuICAgICAgICByZXR1cm4ge1xuICAgICAgICAgIHRpbWVNaWxpc2Vjb25kczogaXRlbS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgICAgZHVyYXRpb25NaWxpc2Vjb25kczogaXRlbS5kdXJhdGlvbk1pbGlzZWNvbmRzLFxuICAgICAgICAgIGJvZHlTaW1pbGFyaXR5OiB1bmRlZmluZWQsXG4gICAgICAgICAgaGFuZFNpbWlsYXJpdHk6IHVuZGVmaW5lZCxcbiAgICAgICAgfTtcbiAgICAgIH0pO1xuICAgIHNlbGVjdGVkUG9zZS5tZXJnZWRUaW1lTWlsaXNlY29uZHMgPVxuICAgICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlWzBdLnRpbWVNaWxpc2Vjb25kcztcbiAgICBzZWxlY3RlZFBvc2UubWVyZ2VkRHVyYXRpb25NaWxpc2Vjb25kcyA9IHRoaXMuc2ltaWxhclBvc2VRdWV1ZS5yZWR1Y2UoXG4gICAgICAoc3VtOiBudW1iZXIsIGl0ZW06IFBvc2VTZXRJdGVtKSA9PiB7XG4gICAgICAgIHJldHVybiBzdW0gKyBpdGVtLmR1cmF0aW9uTWlsaXNlY29uZHM7XG4gICAgICB9LFxuICAgICAgMFxuICAgICk7XG5cbiAgICAvLyDlvZPoqbLjg53jg7zjgrrjgpLjg53jg7zjgrrphY3liJfjgbjov73liqBcbiAgICBpZiAodGhpcy5JU19FTkFCTEVEX1JFTU9WRV9EVVBMSUNBVEVEX1BPU0VTX0ZPUl9BUk9VTkQpIHtcbiAgICAgIHRoaXMucG9zZXMucHVzaChzZWxlY3RlZFBvc2UpO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyDjg4fjg5Djg4PjgrDnlKhcbiAgICAgIHRoaXMucG9zZXMucHVzaCguLi50aGlzLnNpbWlsYXJQb3NlUXVldWUpO1xuICAgIH1cblxuICAgIC8vIOmhnuS8vOODneODvOOCuuOCreODpeODvOOCkuOCr+ODquOColxuICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZSA9IFtdO1xuICB9XG5cbiAgcHJpdmF0ZSByZW1vdmVEdXBsaWNhdGVkUG9zZXMoKTogdm9pZCB7XG4gICAgLy8g5YWo44Od44O844K644KS5q+U6LyD44GX44Gm6aGe5Ly844Od44O844K644KS5YmK6ZmkXG4gICAgY29uc3QgbmV3UG9zZXM6IFBvc2VTZXRJdGVtW10gPSBbXSxcbiAgICAgIHJlbW92ZWRQb3NlczogUG9zZVNldEl0ZW1bXSA9IFtdO1xuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgZHVwbGljYXRlZFBvc2U6IFBvc2VTZXRJdGVtO1xuICAgICAgZm9yIChjb25zdCBpbnNlcnRlZFBvc2Ugb2YgbmV3UG9zZXMpIHtcbiAgICAgICAgY29uc3QgaXNTaW1pbGFyQm9keVBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckJvZHlQb3NlKFxuICAgICAgICAgIHBvc2UuYm9keVZlY3RvcixcbiAgICAgICAgICBpbnNlcnRlZFBvc2UuYm9keVZlY3RvclxuICAgICAgICApO1xuICAgICAgICBjb25zdCBpc1NpbWlsYXJIYW5kUG9zZSA9XG4gICAgICAgICAgcG9zZS5oYW5kVmVjdG9yICYmIGluc2VydGVkUG9zZS5oYW5kVmVjdG9yXG4gICAgICAgICAgICA/IFBvc2VTZXQuaXNTaW1pbGFySGFuZFBvc2UoXG4gICAgICAgICAgICAgICAgcG9zZS5oYW5kVmVjdG9yLFxuICAgICAgICAgICAgICAgIGluc2VydGVkUG9zZS5oYW5kVmVjdG9yLFxuICAgICAgICAgICAgICAgIDAuOVxuICAgICAgICAgICAgICApXG4gICAgICAgICAgICA6IGZhbHNlO1xuXG4gICAgICAgIGlmIChpc1NpbWlsYXJCb2R5UG9zZSAmJiBpc1NpbWlsYXJIYW5kUG9zZSkge1xuICAgICAgICAgIC8vIOi6q+S9k+ODu+aJi+OBqOOCguOBq+mhnuS8vOODneODvOOCuuOBquOCieOBsFxuICAgICAgICAgIGR1cGxpY2F0ZWRQb3NlID0gaW5zZXJ0ZWRQb3NlO1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChkdXBsaWNhdGVkUG9zZSkge1xuICAgICAgICByZW1vdmVkUG9zZXMucHVzaChwb3NlKTtcbiAgICAgICAgaWYgKGR1cGxpY2F0ZWRQb3NlLmRlYnVnLmR1cGxpY2F0ZWRJdGVtcykge1xuICAgICAgICAgIGR1cGxpY2F0ZWRQb3NlLmRlYnVnLmR1cGxpY2F0ZWRJdGVtcy5wdXNoKHtcbiAgICAgICAgICAgIHRpbWVNaWxpc2Vjb25kczogcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBwb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMsXG4gICAgICAgICAgICBib2R5U2ltaWxhcml0eTogdW5kZWZpbmVkLFxuICAgICAgICAgICAgaGFuZFNpbWlsYXJpdHk6IHVuZGVmaW5lZCxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfVxuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cblxuICAgICAgbmV3UG9zZXMucHVzaChwb3NlKTtcbiAgICB9XG5cbiAgICBjb25zb2xlLmluZm8oXG4gICAgICBgW1Bvc2VTZXRdIHJlbW92ZUR1cGxpY2F0ZWRQb3NlcyAtIFJlZHVjZWQgJHt0aGlzLnBvc2VzLmxlbmd0aH0gcG9zZXMgLT4gJHtuZXdQb3Nlcy5sZW5ndGh9IHBvc2VzYCxcbiAgICAgIHtcbiAgICAgICAgcmVtb3ZlZDogcmVtb3ZlZFBvc2VzLFxuICAgICAgICBrZWVwZWQ6IG5ld1Bvc2VzLFxuICAgICAgfVxuICAgICk7XG4gICAgdGhpcy5wb3NlcyA9IG5ld1Bvc2VzO1xuICB9XG5cbiAgcHJpdmF0ZSBnZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKElNQUdFX01JTUU6IHN0cmluZykge1xuICAgIHN3aXRjaCAoSU1BR0VfTUlNRSkge1xuICAgICAgY2FzZSAnaW1hZ2UvcG5nJzpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgICAgY2FzZSAnaW1hZ2UvanBlZyc6XG4gICAgICAgIHJldHVybiAnanBnJztcbiAgICAgIGNhc2UgJ2ltYWdlL3dlYnAnOlxuICAgICAgICByZXR1cm4gJ3dlYnAnO1xuICAgICAgZGVmYXVsdDpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgIH1cbiAgfVxufVxuIl19