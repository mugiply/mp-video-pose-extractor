import { POSE_LANDMARKS } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
// @ts-ignore
import cosSimilarityA from 'cos-similarity';
// @ts-ignore
import * as cosSimilarityB from 'cos-similarity';
import { ImageTrimmer } from './internals/image-trimmer';
export class PoseSet {
    constructor() {
        this.poses = [];
        this.isFinalized = false;
        // ポーズを追加するためのキュー
        this.similarPoseQueue = [];
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
        if (this.poses.length === 0) {
            this.videoMetadata.firstPoseDetectedTime = videoTimeMiliseconds;
        }
        if (results.poseLandmarks === undefined) {
            console.debug(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get poseLandmarks`, results);
            return;
        }
        const poseLandmarksWithWorldCoordinate = results.ea
            ? results.ea
            : [];
        if (poseLandmarksWithWorldCoordinate.length === 0) {
            console.debug(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the pose with the world coordinate`, results);
            return;
        }
        const bodyVector = PoseSet.getBodyVector(poseLandmarksWithWorldCoordinate);
        if (!bodyVector) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the body vector`, poseLandmarksWithWorldCoordinate);
            return;
        }
        if (results.leftHandLandmarks === undefined &&
            results.rightHandLandmarks === undefined) {
            console.debug(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the hand landmarks`, results);
        }
        else if (results.leftHandLandmarks === undefined) {
            console.debug(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the left hand landmarks`, results);
        }
        else if (results.rightHandLandmarks === undefined) {
            console.debug(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the right hand landmarks`, results);
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
    async finalize(isRemoveDuplicate = true) {
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
        if (isRemoveDuplicate) {
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
            leftWristToLeftElbow: PoseSet.getCosSimilarity(bodyVectorA.leftWristToLeftElbow, bodyVectorB.leftWristToLeftElbow),
            leftElbowToLeftShoulder: PoseSet.getCosSimilarity(bodyVectorA.leftElbowToLeftShoulder, bodyVectorB.leftElbowToLeftShoulder),
            rightWristToRightElbow: PoseSet.getCosSimilarity(bodyVectorA.rightWristToRightElbow, bodyVectorB.rightWristToRightElbow),
            rightElbowToRightShoulder: PoseSet.getCosSimilarity(bodyVectorA.rightElbowToRightShoulder, bodyVectorB.rightElbowToRightShoulder),
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
                rightThumbTipToFirstJoint: PoseSet.getCosSimilarity(handVectorA.rightThumbTipToFirstJoint, handVectorB.rightThumbTipToFirstJoint),
                rightThumbFirstJointToSecondJoint: PoseSet.getCosSimilarity(handVectorA.rightThumbFirstJointToSecondJoint, handVectorB.rightThumbFirstJointToSecondJoint),
                // 右手 - 人差し指
                rightIndexFingerTipToFirstJoint: PoseSet.getCosSimilarity(handVectorA.rightIndexFingerTipToFirstJoint, handVectorB.rightIndexFingerTipToFirstJoint),
                rightIndexFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(handVectorA.rightIndexFingerFirstJointToSecondJoint, handVectorB.rightIndexFingerFirstJointToSecondJoint),
                // 右手 - 中指
                rightMiddleFingerTipToFirstJoint: PoseSet.getCosSimilarity(handVectorA.rightMiddleFingerTipToFirstJoint, handVectorB.rightMiddleFingerTipToFirstJoint),
                rightMiddleFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(handVectorA.rightMiddleFingerFirstJointToSecondJoint, handVectorB.rightMiddleFingerFirstJointToSecondJoint),
                // 右手 - 薬指
                rightRingFingerTipToFirstJoint: PoseSet.getCosSimilarity(handVectorA.rightRingFingerTipToFirstJoint, handVectorB.rightRingFingerFirstJointToSecondJoint),
                rightRingFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(handVectorA.rightRingFingerFirstJointToSecondJoint, handVectorB.rightRingFingerFirstJointToSecondJoint),
                // 右手 - 小指
                rightPinkyFingerTipToFirstJoint: PoseSet.getCosSimilarity(handVectorA.rightPinkyFingerTipToFirstJoint, handVectorB.rightPinkyFingerTipToFirstJoint),
                rightPinkyFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(handVectorA.rightPinkyFingerFirstJointToSecondJoint, handVectorB.rightPinkyFingerFirstJointToSecondJoint),
            };
        const cosSimilaritiesLeftHand = handVectorA.leftThumbFirstJointToSecondJoint === null ||
            handVectorB.leftThumbFirstJointToSecondJoint === null
            ? undefined
            : {
                // 左手 - 親指
                leftThumbTipToFirstJoint: PoseSet.getCosSimilarity(handVectorA.leftThumbTipToFirstJoint, handVectorB.leftThumbTipToFirstJoint),
                leftThumbFirstJointToSecondJoint: PoseSet.getCosSimilarity(handVectorA.leftThumbFirstJointToSecondJoint, handVectorB.leftThumbFirstJointToSecondJoint),
                // 左手 - 人差し指
                leftIndexFingerTipToFirstJoint: PoseSet.getCosSimilarity(handVectorA.leftIndexFingerTipToFirstJoint, handVectorB.leftIndexFingerTipToFirstJoint),
                leftIndexFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(handVectorA.leftIndexFingerFirstJointToSecondJoint, handVectorB.leftIndexFingerFirstJointToSecondJoint),
                // 左手 - 中指
                leftMiddleFingerTipToFirstJoint: PoseSet.getCosSimilarity(handVectorA.leftMiddleFingerTipToFirstJoint, handVectorB.leftMiddleFingerTipToFirstJoint),
                leftMiddleFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(handVectorA.leftMiddleFingerFirstJointToSecondJoint, handVectorB.leftMiddleFingerFirstJointToSecondJoint),
                // 左手 - 薬指
                leftRingFingerTipToFirstJoint: PoseSet.getCosSimilarity(handVectorA.leftRingFingerTipToFirstJoint, handVectorB.leftRingFingerTipToFirstJoint),
                leftRingFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(handVectorA.leftRingFingerFirstJointToSecondJoint, handVectorB.leftRingFingerFirstJointToSecondJoint),
                // 左手 - 小指
                leftPinkyFingerTipToFirstJoint: PoseSet.getCosSimilarity(handVectorA.leftPinkyFingerTipToFirstJoint, handVectorB.leftPinkyFingerTipToFirstJoint),
                leftPinkyFingerFirstJointToSecondJoint: PoseSet.getCosSimilarity(handVectorA.leftPinkyFingerFirstJointToSecondJoint, handVectorB.leftPinkyFingerFirstJointToSecondJoint),
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
    static getCosSimilarity(a, b) {
        if (cosSimilarityA) {
            return cosSimilarityA(a, b);
        }
        return cosSimilarityB(a, b);
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1zZXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi9wcm9qZWN0cy9uZ3gtbXAtcG9zZS1leHRyYWN0b3Ivc3JjL2xpYi9jbGFzc2VzL3Bvc2Utc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxjQUFjLEVBQVcsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssS0FBSyxNQUFNLE9BQU8sQ0FBQztBQU0vQixhQUFhO0FBQ2IsT0FBTyxjQUFjLE1BQU0sZ0JBQWdCLENBQUM7QUFDNUMsYUFBYTtBQUNiLE9BQU8sS0FBSyxjQUFjLE1BQU0sZ0JBQWdCLENBQUM7QUFHakQsT0FBTyxFQUFFLFlBQVksRUFBRSxNQUFNLDJCQUEyQixDQUFDO0FBR3pELE1BQU0sT0FBTyxPQUFPO0lBOEVsQjtRQXBFTyxVQUFLLEdBQWtCLEVBQUUsQ0FBQztRQUMxQixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQThDckMsaUJBQWlCO1FBQ1QscUJBQWdCLEdBQWtCLEVBQUUsQ0FBQztRQUU3Qyx1QkFBdUI7UUFDTixrREFBNkMsR0FBRyxJQUFJLENBQUM7UUFFdEUsYUFBYTtRQUNJLGdCQUFXLEdBQVcsSUFBSSxDQUFDO1FBQzNCLGVBQVUsR0FDekIsWUFBWSxDQUFDO1FBQ0Usa0JBQWEsR0FBRyxHQUFHLENBQUM7UUFFckMsVUFBVTtRQUNPLGdDQUEyQixHQUFHLFNBQVMsQ0FBQztRQUN4Qyx5Q0FBb0MsR0FBRyxFQUFFLENBQUM7UUFFM0QsV0FBVztRQUNNLHVDQUFrQyxHQUFHLFNBQVMsQ0FBQztRQUMvQyx1Q0FBa0MsR0FBRyxXQUFXLENBQUM7UUFDakQsNENBQXVDLEdBQUcsR0FBRyxDQUFDO1FBRzdELElBQUksQ0FBQyxhQUFhLEdBQUc7WUFDbkIsSUFBSSxFQUFFLEVBQUU7WUFDUixLQUFLLEVBQUUsQ0FBQztZQUNSLE1BQU0sRUFBRSxDQUFDO1lBQ1QsUUFBUSxFQUFFLENBQUM7WUFDWCxx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Q3hCLDJCQUEyQjtJQUMzQixLQUFLO0lBQ0wsc0JBQXNCO0lBQ3RCLHlCQUF5QjtDQUMxQixDQUFDO0FBRUYsa0JBQWtCO0FBQ0ssNEJBQW9CLEdBQUc7SUFDNUMsVUFBVTtJQUNWLDJCQUEyQjtJQUMzQixtQ0FBbUM7SUFDbkMsWUFBWTtJQUNaLGlDQUFpQztJQUNqQyx5Q0FBeUM7SUFDekMsVUFBVTtJQUNWLGtDQUFrQztJQUNsQywwQ0FBMEM7SUFDMUMsVUFBVTtJQUNWLGdDQUFnQztJQUNoQyx3Q0FBd0M7SUFDeEMsVUFBVTtJQUNWLGlDQUFpQztJQUNqQyx5Q0FBeUM7SUFDekMsVUFBVTtJQUNWLDBCQUEwQjtJQUMxQixrQ0FBa0M7SUFDbEMsWUFBWTtJQUNaLGdDQUFnQztJQUNoQyx3Q0FBd0M7SUFDeEMsVUFBVTtJQUNWLGlDQUFpQztJQUNqQyx5Q0FBeUM7SUFDekMsVUFBVTtJQUNWLCtCQUErQjtJQUMvQix1Q0FBdUM7SUFDdkMsVUFBVTtJQUNWLGdDQUFnQztJQUNoQyx3Q0FBd0M7Q0FDekMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IFBPU0VfTEFORE1BUktTLCBSZXN1bHRzIH0gZnJvbSAnQG1lZGlhcGlwZS9ob2xpc3RpYyc7XG5pbXBvcnQgKiBhcyBKU1ppcCBmcm9tICdqc3ppcCc7XG5pbXBvcnQgeyBQb3NlU2V0SXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1zZXQtaXRlbSc7XG5pbXBvcnQgeyBQb3NlU2V0SnNvbiB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1zZXQtanNvbic7XG5pbXBvcnQgeyBQb3NlU2V0SnNvbkl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWpzb24taXRlbSc7XG5pbXBvcnQgeyBCb2R5VmVjdG9yIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9ib2R5LXZlY3Rvcic7XG5cbi8vIEB0cy1pZ25vcmVcbmltcG9ydCBjb3NTaW1pbGFyaXR5QSBmcm9tICdjb3Mtc2ltaWxhcml0eSc7XG4vLyBAdHMtaWdub3JlXG5pbXBvcnQgKiBhcyBjb3NTaW1pbGFyaXR5QiBmcm9tICdjb3Mtc2ltaWxhcml0eSc7XG5cbmltcG9ydCB7IFNpbWlsYXJQb3NlSXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvc2ltaWxhci1wb3NlLWl0ZW0nO1xuaW1wb3J0IHsgSW1hZ2VUcmltbWVyIH0gZnJvbSAnLi9pbnRlcm5hbHMvaW1hZ2UtdHJpbW1lcic7XG5pbXBvcnQgeyBIYW5kVmVjdG9yIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9oYW5kLXZlY3Rvcic7XG5cbmV4cG9ydCBjbGFzcyBQb3NlU2V0IHtcbiAgcHVibGljIGdlbmVyYXRvcj86IHN0cmluZztcbiAgcHVibGljIHZlcnNpb24/OiBudW1iZXI7XG4gIHByaXZhdGUgdmlkZW9NZXRhZGF0YSE6IHtcbiAgICBuYW1lOiBzdHJpbmc7XG4gICAgd2lkdGg6IG51bWJlcjtcbiAgICBoZWlnaHQ6IG51bWJlcjtcbiAgICBkdXJhdGlvbjogbnVtYmVyO1xuICAgIGZpcnN0UG9zZURldGVjdGVkVGltZTogbnVtYmVyO1xuICB9O1xuICBwdWJsaWMgcG9zZXM6IFBvc2VTZXRJdGVtW10gPSBbXTtcbiAgcHVibGljIGlzRmluYWxpemVkPzogYm9vbGVhbiA9IGZhbHNlO1xuXG4gIC8vIEJvZHlWZWN0b3Ig44Gu44Kt44O85ZCNXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgQk9EWV9WRUNUT1JfTUFQUElOR1MgPSBbXG4gICAgLy8g5Y+z6IWVXG4gICAgJ3JpZ2h0V3Jpc3RUb1JpZ2h0RWxib3cnLFxuICAgICdyaWdodEVsYm93VG9SaWdodFNob3VsZGVyJyxcbiAgICAvLyDlt6bohZVcbiAgICAnbGVmdFdyaXN0VG9MZWZ0RWxib3cnLFxuICAgICdsZWZ0RWxib3dUb0xlZnRTaG91bGRlcicsXG4gIF07XG5cbiAgLy8gSGFuZFZlY3RvciDjga7jgq3jg7zlkI1cbiAgcHVibGljIHN0YXRpYyByZWFkb25seSBIQU5EX1ZFQ1RPUl9NQVBQSU5HUyA9IFtcbiAgICAvLyDlj7PmiYsgLSDopqrmjIdcbiAgICAncmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ3JpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5Y+z5omLIC0g5Lq65beu44GX5oyHXG4gICAgJ3JpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOS4reaMh1xuICAgICdyaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ3JpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOiWrOaMh1xuICAgICdyaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5Y+z5omLIC0g5bCP5oyHXG4gICAgJ3JpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOimquaMh1xuICAgICdsZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g5Lq65beu44GX5oyHXG4gICAgJ2xlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlt6bmiYsgLSDkuK3mjIdcbiAgICAnbGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g6Jas5oyHXG4gICAgJ2xlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAnbGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g5bCP5oyHXG4gICAgJ2xlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgXTtcblxuICAvLyDjg53jg7zjgrrjgpLov73liqDjgZnjgovjgZ/jgoHjga7jgq3jg6Xjg7xcbiAgcHJpdmF0ZSBzaW1pbGFyUG9zZVF1ZXVlOiBQb3NlU2V0SXRlbVtdID0gW107XG5cbiAgLy8g6aGe5Ly844Od44O844K644Gu6Zmk5Y67IC0g5ZCE44Od44O844K644Gu5YmN5b6M44GL44KJXG4gIHByaXZhdGUgcmVhZG9ubHkgSVNfRU5BQkxFRF9SRU1PVkVfRFVQTElDQVRFRF9QT1NFU19GT1JfQVJPVU5EID0gdHJ1ZTtcblxuICAvLyDnlLvlg4/mm7jjgY3lh7rjgZfmmYLjga7oqK3lrppcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9XSURUSDogbnVtYmVyID0gMTA4MDtcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9NSU1FOiAnaW1hZ2UvanBlZycgfCAnaW1hZ2UvcG5nJyB8ICdpbWFnZS93ZWJwJyA9XG4gICAgJ2ltYWdlL3dlYnAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX1FVQUxJVFkgPSAwLjg7XG5cbiAgLy8g55S75YOP44Gu5L2Z55m96Zmk5Y67XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0NPTE9SID0gJyMwMDAwMDAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX01BUkdJTl9UUklNTUlOR19ESUZGX1RIUkVTSE9MRCA9IDUwO1xuXG4gIC8vIOeUu+WDj+OBruiDjOaZr+iJsue9ruaPm1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9TUkNfQ09MT1IgPSAnIzAxNkFGRCc7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RTVF9DT0xPUiA9ICcjRkZGRkZGMDAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9ESUZGX1RIUkVTSE9MRCA9IDEzMDtcblxuICBjb25zdHJ1Y3RvcigpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSB7XG4gICAgICBuYW1lOiAnJyxcbiAgICAgIHdpZHRoOiAwLFxuICAgICAgaGVpZ2h0OiAwLFxuICAgICAgZHVyYXRpb246IDAsXG4gICAgICBmaXJzdFBvc2VEZXRlY3RlZFRpbWU6IDAsXG4gICAgfTtcbiAgfVxuXG4gIGdldFZpZGVvTmFtZSgpIHtcbiAgICByZXR1cm4gdGhpcy52aWRlb01ldGFkYXRhLm5hbWU7XG4gIH1cblxuICBzZXRWaWRlb05hbWUodmlkZW9OYW1lOiBzdHJpbmcpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEubmFtZSA9IHZpZGVvTmFtZTtcbiAgfVxuXG4gIHNldFZpZGVvTWV0YURhdGEod2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIsIGR1cmF0aW9uOiBudW1iZXIpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEud2lkdGggPSB3aWR0aDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuaGVpZ2h0ID0gaGVpZ2h0O1xuICAgIHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbiA9IGR1cmF0aW9uO1xuICB9XG5cbiAgLyoqXG4gICAqIOODneODvOOCuuaVsOOBruWPluW+l1xuICAgKiBAcmV0dXJuc1xuICAgKi9cbiAgZ2V0TnVtYmVyT2ZQb3NlcygpOiBudW1iZXIge1xuICAgIGlmICh0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpIHJldHVybiAtMTtcbiAgICByZXR1cm4gdGhpcy5wb3Nlcy5sZW5ndGg7XG4gIH1cblxuICAvKipcbiAgICog5YWo44Od44O844K644Gu5Y+W5b6XXG4gICAqIEByZXR1cm5zIOWFqOOBpuOBruODneODvOOCulxuICAgKi9cbiAgZ2V0UG9zZXMoKTogUG9zZVNldEl0ZW1bXSB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIFtdO1xuICAgIHJldHVybiB0aGlzLnBvc2VzO1xuICB9XG5cbiAgLyoqXG4gICAqIOaMh+WumuOBleOCjOOBn+aZgumWk+OBq+OCiOOCi+ODneODvOOCuuOBruWPluW+l1xuICAgKiBAcGFyYW0gdGltZU1pbGlzZWNvbmRzIOODneODvOOCuuOBruaZgumWkyAo44Of44Oq56eSKVxuICAgKiBAcmV0dXJucyDjg53jg7zjgrpcbiAgICovXG4gIGdldFBvc2VCeVRpbWUodGltZU1pbGlzZWNvbmRzOiBudW1iZXIpOiBQb3NlU2V0SXRlbSB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICByZXR1cm4gdGhpcy5wb3Nlcy5maW5kKChwb3NlKSA9PiBwb3NlLnRpbWVNaWxpc2Vjb25kcyA9PT0gdGltZU1pbGlzZWNvbmRzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiDjg53jg7zjgrrjga7ov73liqBcbiAgICovXG4gIHB1c2hQb3NlKFxuICAgIHZpZGVvVGltZU1pbGlzZWNvbmRzOiBudW1iZXIsXG4gICAgZnJhbWVJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICBwb3NlSW1hZ2VEYXRhVXJsOiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gICAgZmFjZUZyYW1lSW1hZ2VEYXRhVXJsOiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gICAgcmVzdWx0czogUmVzdWx0c1xuICApOiBQb3NlU2V0SXRlbSB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKHRoaXMucG9zZXMubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZmlyc3RQb3NlRGV0ZWN0ZWRUaW1lID0gdmlkZW9UaW1lTWlsaXNlY29uZHM7XG4gICAgfVxuXG4gICAgaWYgKHJlc3VsdHMucG9zZUxhbmRtYXJrcyA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHBvc2VMYW5kbWFya3NgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlOiBhbnlbXSA9IChyZXN1bHRzIGFzIGFueSkuZWFcbiAgICAgID8gKHJlc3VsdHMgYXMgYW55KS5lYVxuICAgICAgOiBbXTtcbiAgICBpZiAocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubGVuZ3RoID09PSAwKSB7XG4gICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBwb3NlIHdpdGggdGhlIHdvcmxkIGNvb3JkaW5hdGVgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IGJvZHlWZWN0b3IgPSBQb3NlU2V0LmdldEJvZHlWZWN0b3IocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUpO1xuICAgIGlmICghYm9keVZlY3Rvcikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBib2R5IHZlY3RvcmAsXG4gICAgICAgIHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGlmIChcbiAgICAgIHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCAmJlxuICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZFxuICAgICkge1xuICAgICAgY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgaGFuZCBsYW5kbWFya3NgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH0gZWxzZSBpZiAocmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBsZWZ0IGhhbmQgbGFuZG1hcmtzYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHtcbiAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIHJpZ2h0IGhhbmQgbGFuZG1hcmtzYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICB9XG5cbiAgICBjb25zdCBoYW5kVmVjdG9yID0gUG9zZVNldC5nZXRIYW5kVmVjdG9yKFxuICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyxcbiAgICAgIHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzXG4gICAgKTtcbiAgICBpZiAoIWhhbmRWZWN0b3IpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgaGFuZCB2ZWN0b3JgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2U6IFBvc2VTZXRJdGVtID0ge1xuICAgICAgdGltZU1pbGlzZWNvbmRzOiB2aWRlb1RpbWVNaWxpc2Vjb25kcyxcbiAgICAgIGR1cmF0aW9uTWlsaXNlY29uZHM6IC0xLFxuICAgICAgcG9zZTogcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubWFwKCh3b3JsZENvb3JkaW5hdGVMYW5kbWFyaykgPT4ge1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLngsXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsueSxcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay56LFxuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLnZpc2liaWxpdHksXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIGxlZnRIYW5kOiByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzPy5tYXAoKG5vcm1hbGl6ZWRMYW5kbWFyaykgPT4ge1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay54LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay55LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay56LFxuICAgICAgICBdO1xuICAgICAgfSksXG4gICAgICByaWdodEhhbmQ6IHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzPy5tYXAoKG5vcm1hbGl6ZWRMYW5kbWFyaykgPT4ge1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay54LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay55LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay56LFxuICAgICAgICBdO1xuICAgICAgfSksXG4gICAgICBib2R5VmVjdG9yOiBib2R5VmVjdG9yLFxuICAgICAgaGFuZFZlY3RvcjogaGFuZFZlY3RvcixcbiAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiBmcmFtZUltYWdlRGF0YVVybCxcbiAgICAgIHBvc2VJbWFnZURhdGFVcmw6IHBvc2VJbWFnZURhdGFVcmwsXG4gICAgICBmYWNlRnJhbWVJbWFnZURhdGFVcmw6IGZhY2VGcmFtZUltYWdlRGF0YVVybCxcbiAgICAgIGV4dGVuZGVkRGF0YToge30sXG4gICAgICBkZWJ1Zzoge1xuICAgICAgICBkdXBsaWNhdGVkSXRlbXM6IFtdLFxuICAgICAgfSxcbiAgICAgIG1lcmdlZFRpbWVNaWxpc2Vjb25kczogLTEsXG4gICAgICBtZXJnZWREdXJhdGlvbk1pbGlzZWNvbmRzOiAtMSxcbiAgICB9O1xuXG4gICAgbGV0IGxhc3RQb3NlO1xuICAgIGlmICh0aGlzLnBvc2VzLmxlbmd0aCA9PT0gMCAmJiAxIDw9IHRoaXMuc2ltaWxhclBvc2VRdWV1ZS5sZW5ndGgpIHtcbiAgICAgIC8vIOmhnuS8vOODneODvOOCuuOCreODpeODvOOBi+OCieacgOW+jOOBruODneODvOOCuuOCkuWPluW+l1xuICAgICAgbGFzdFBvc2UgPSB0aGlzLnNpbWlsYXJQb3NlUXVldWVbdGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCAtIDFdO1xuICAgIH0gZWxzZSBpZiAoMSA8PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgLy8g44Od44O844K66YWN5YiX44GL44KJ5pyA5b6M44Gu44Od44O844K644KS5Y+W5b6XXG4gICAgICBsYXN0UG9zZSA9IHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXTtcbiAgICB9XG5cbiAgICBpZiAobGFzdFBvc2UpIHtcbiAgICAgIC8vIOacgOW+jOOBruODneODvOOCuuOBjOOBguOCjOOBsOOAgemhnuS8vOODneODvOOCuuOBi+OBqeOBhuOBi+OCkuavlOi8g1xuICAgICAgY29uc3QgaXNTaW1pbGFyQm9keVBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckJvZHlQb3NlKFxuICAgICAgICBwb3NlLmJvZHlWZWN0b3IsXG4gICAgICAgIGxhc3RQb3NlLmJvZHlWZWN0b3JcbiAgICAgICk7XG5cbiAgICAgIGxldCBpc1NpbWlsYXJIYW5kUG9zZSA9IHRydWU7XG4gICAgICBpZiAobGFzdFBvc2UuaGFuZFZlY3RvciAmJiBwb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgaXNTaW1pbGFySGFuZFBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckhhbmRQb3NlKFxuICAgICAgICAgIHBvc2UuaGFuZFZlY3RvcixcbiAgICAgICAgICBsYXN0UG9zZS5oYW5kVmVjdG9yXG4gICAgICAgICk7XG4gICAgICB9IGVsc2UgaWYgKCFsYXN0UG9zZS5oYW5kVmVjdG9yICYmIHBvc2UuaGFuZFZlY3Rvcikge1xuICAgICAgICBpc1NpbWlsYXJIYW5kUG9zZSA9IGZhbHNlO1xuICAgICAgfVxuXG4gICAgICBpZiAoIWlzU2ltaWxhckJvZHlQb3NlIHx8ICFpc1NpbWlsYXJIYW5kUG9zZSkge1xuICAgICAgICAvLyDouqvkvZPjg7vmiYvjga7jgYTjgZrjgozjgYvjgYzliY3jga7jg53jg7zjgrrjgajpoZ7kvLzjgZfjgabjgYTjgarjgYTjgarjgonjgbDjgIHpoZ7kvLzjg53jg7zjgrrjgq3jg6Xjg7zjgpLlh6bnkIbjgZfjgabjgIHjg53jg7zjgrrphY3liJfjgbjov73liqBcbiAgICAgICAgdGhpcy5wdXNoUG9zZUZyb21TaW1pbGFyUG9zZVF1ZXVlKHBvc2UudGltZU1pbGlzZWNvbmRzKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyDpoZ7kvLzjg53jg7zjgrrjgq3jg6Xjg7zjgbjov73liqBcbiAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWUucHVzaChwb3NlKTtcblxuICAgIHJldHVybiBwb3NlO1xuICB9XG5cbiAgLyoqXG4gICAqIOODneODvOOCuuOBrumFjeWIl+OBi+OCieODneODvOOCuuOBjOaxuuOBvuOBo+OBpuOBhOOCi+eerOmWk+OCkuWPluW+l1xuICAgKiBAcGFyYW0gcG9zZXMg44Od44O844K644Gu6YWN5YiXXG4gICAqIEByZXR1cm5zIOODneODvOOCuuOBjOaxuuOBvuOBo+OBpuOBhOOCi+eerOmWk1xuICAgKi9cbiAgc3RhdGljIGdldFN1aXRhYmxlUG9zZUJ5UG9zZXMocG9zZXM6IFBvc2VTZXRJdGVtW10pOiBQb3NlU2V0SXRlbSB7XG4gICAgaWYgKHBvc2VzLmxlbmd0aCA9PT0gMCkgcmV0dXJuIG51bGw7XG4gICAgaWYgKHBvc2VzLmxlbmd0aCA9PT0gMSkge1xuICAgICAgcmV0dXJuIHBvc2VzWzFdO1xuICAgIH1cblxuICAgIC8vIOWQhOaomeacrOODneODvOOCuuOBlOOBqOOBrumhnuS8vOW6puOCkuWIneacn+WMllxuICAgIGNvbnN0IHNpbWlsYXJpdGllc09mUG9zZXM6IHtcbiAgICAgIFtrZXk6IG51bWJlcl06IHtcbiAgICAgICAgaGFuZFNpbWlsYXJpdHk6IG51bWJlcjtcbiAgICAgICAgYm9keVNpbWlsYXJpdHk6IG51bWJlcjtcbiAgICAgIH1bXTtcbiAgICB9ID0ge307XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBwb3Nlcy5sZW5ndGg7IGkrKykge1xuICAgICAgc2ltaWxhcml0aWVzT2ZQb3Nlc1twb3Nlc1tpXS50aW1lTWlsaXNlY29uZHNdID0gcG9zZXMubWFwKFxuICAgICAgICAocG9zZTogUG9zZVNldEl0ZW0pID0+IHtcbiAgICAgICAgICByZXR1cm4ge1xuICAgICAgICAgICAgaGFuZFNpbWlsYXJpdHk6IDAsXG4gICAgICAgICAgICBib2R5U2ltaWxhcml0eTogMCxcbiAgICAgICAgICB9O1xuICAgICAgICB9XG4gICAgICApO1xuICAgIH1cblxuICAgIC8vIOWQhOaomeacrOODneODvOOCuuOBlOOBqOOBrumhnuS8vOW6puOCkuioiOeul1xuICAgIGZvciAobGV0IHNhbXBsZVBvc2Ugb2YgcG9zZXMpIHtcbiAgICAgIGxldCBoYW5kU2ltaWxhcml0eTogbnVtYmVyO1xuXG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHBvc2VzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIGNvbnN0IHBvc2UgPSBwb3Nlc1tpXTtcbiAgICAgICAgaWYgKHBvc2UuaGFuZFZlY3RvciAmJiBzYW1wbGVQb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgICBoYW5kU2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0SGFuZFNpbWlsYXJpdHkoXG4gICAgICAgICAgICBwb3NlLmhhbmRWZWN0b3IsXG4gICAgICAgICAgICBzYW1wbGVQb3NlLmhhbmRWZWN0b3JcbiAgICAgICAgICApO1xuICAgICAgICB9XG5cbiAgICAgICAgbGV0IGJvZHlTaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRCb2R5UG9zZVNpbWlsYXJpdHkoXG4gICAgICAgICAgcG9zZS5ib2R5VmVjdG9yLFxuICAgICAgICAgIHNhbXBsZVBvc2UuYm9keVZlY3RvclxuICAgICAgICApO1xuXG4gICAgICAgIHNpbWlsYXJpdGllc09mUG9zZXNbc2FtcGxlUG9zZS50aW1lTWlsaXNlY29uZHNdW2ldID0ge1xuICAgICAgICAgIGhhbmRTaW1pbGFyaXR5OiBoYW5kU2ltaWxhcml0eSA/PyAwLFxuICAgICAgICAgIGJvZHlTaW1pbGFyaXR5LFxuICAgICAgICB9O1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIOmhnuS8vOW6puOBrumrmOOBhOODleODrOODvOODoOOBjOWkmuOBi+OBo+OBn+ODneODvOOCuuOCkumBuOaKnlxuICAgIGNvbnN0IHNpbWlsYXJpdGllc09mU2FtcGxlUG9zZXMgPSBwb3Nlcy5tYXAoKHBvc2U6IFBvc2VTZXRJdGVtKSA9PiB7XG4gICAgICByZXR1cm4gc2ltaWxhcml0aWVzT2ZQb3Nlc1twb3NlLnRpbWVNaWxpc2Vjb25kc10ucmVkdWNlKFxuICAgICAgICAoXG4gICAgICAgICAgcHJldjogbnVtYmVyLFxuICAgICAgICAgIGN1cnJlbnQ6IHsgaGFuZFNpbWlsYXJpdHk6IG51bWJlcjsgYm9keVNpbWlsYXJpdHk6IG51bWJlciB9XG4gICAgICAgICkgPT4ge1xuICAgICAgICAgIHJldHVybiBwcmV2ICsgY3VycmVudC5oYW5kU2ltaWxhcml0eSArIGN1cnJlbnQuYm9keVNpbWlsYXJpdHk7XG4gICAgICAgIH0sXG4gICAgICAgIDBcbiAgICAgICk7XG4gICAgfSk7XG4gICAgY29uc3QgbWF4U2ltaWxhcml0eSA9IE1hdGgubWF4KC4uLnNpbWlsYXJpdGllc09mU2FtcGxlUG9zZXMpO1xuICAgIGNvbnN0IG1heFNpbWlsYXJpdHlJbmRleCA9IHNpbWlsYXJpdGllc09mU2FtcGxlUG9zZXMuaW5kZXhPZihtYXhTaW1pbGFyaXR5KTtcbiAgICBjb25zdCBzZWxlY3RlZFBvc2UgPSBwb3Nlc1ttYXhTaW1pbGFyaXR5SW5kZXhdO1xuICAgIGlmICghc2VsZWN0ZWRQb3NlKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gZ2V0U3VpdGFibGVQb3NlQnlQb3Nlc2AsXG4gICAgICAgIHNpbWlsYXJpdGllc09mU2FtcGxlUG9zZXMsXG4gICAgICAgIG1heFNpbWlsYXJpdHksXG4gICAgICAgIG1heFNpbWlsYXJpdHlJbmRleFxuICAgICAgKTtcbiAgICB9XG5cbiAgICBjb25zb2xlLmRlYnVnKGBbUG9zZVNldF0gZ2V0U3VpdGFibGVQb3NlQnlQb3Nlc2AsIHtcbiAgICAgIHNlbGVjdGVkOiBzZWxlY3RlZFBvc2UsXG4gICAgICB1bnNlbGVjdGVkOiBwb3Nlcy5maWx0ZXIoKHBvc2U6IFBvc2VTZXRJdGVtKSA9PiB7XG4gICAgICAgIHJldHVybiBwb3NlLnRpbWVNaWxpc2Vjb25kcyAhPT0gc2VsZWN0ZWRQb3NlLnRpbWVNaWxpc2Vjb25kcztcbiAgICAgIH0pLFxuICAgIH0pO1xuICAgIHJldHVybiBzZWxlY3RlZFBvc2U7XG4gIH1cblxuICAvKipcbiAgICog5pyA57WC5Yem55CGXG4gICAqICjph43opIfjgZfjgZ/jg53jg7zjgrrjga7pmaTljrvjgIHnlLvlg4/jga7jg57jg7zjgrjjg7PpmaTljrvjgarjgakpXG4gICAqL1xuICBhc3luYyBmaW5hbGl6ZShpc1JlbW92ZUR1cGxpY2F0ZTogYm9vbGVhbiA9IHRydWUpIHtcbiAgICBpZiAodGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCA+IDApIHtcbiAgICAgIC8vIOmhnuS8vOODneODvOOCuuOCreODpeODvOOBq+ODneODvOOCuuOBjOaui+OBo+OBpuOBhOOCi+WgtOWQiOOAgeacgOmBqeOBquODneODvOOCuuOCkumBuOaKnuOBl+OBpuODneODvOOCuumFjeWIl+OBuOi/veWKoFxuICAgICAgdGhpcy5wdXNoUG9zZUZyb21TaW1pbGFyUG9zZVF1ZXVlKHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbik7XG4gICAgfVxuXG4gICAgaWYgKDAgPT0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIC8vIOODneODvOOCuuOBjOS4gOOBpOOCguOBquOBhOWgtOWQiOOAgeWHpueQhuOCkue1guS6hlxuICAgICAgdGhpcy5pc0ZpbmFsaXplZCA9IHRydWU7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgLy8g44Od44O844K644Gu5oyB57aa5pmC6ZaT44KS6Kit5a6aXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLnBvc2VzLmxlbmd0aCAtIDE7IGkrKykge1xuICAgICAgaWYgKHRoaXMucG9zZXNbaV0uZHVyYXRpb25NaWxpc2Vjb25kcyAhPT0gLTEpIGNvbnRpbnVlO1xuICAgICAgdGhpcy5wb3Nlc1tpXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgdGhpcy5wb3Nlc1tpICsgMV0udGltZU1pbGlzZWNvbmRzIC0gdGhpcy5wb3Nlc1tpXS50aW1lTWlsaXNlY29uZHM7XG4gICAgfVxuICAgIHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgIHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbiAtXG4gICAgICB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV0udGltZU1pbGlzZWNvbmRzO1xuXG4gICAgLy8g5YWo5L2T44GL44KJ6YeN6KSH44Od44O844K644KS6Zmk5Y67XG4gICAgaWYgKGlzUmVtb3ZlRHVwbGljYXRlKSB7XG4gICAgICB0aGlzLnJlbW92ZUR1cGxpY2F0ZWRQb3NlcygpO1xuICAgIH1cblxuICAgIC8vIOacgOWIneOBruODneODvOOCuuOCkumZpOWOu1xuICAgIHRoaXMucG9zZXMuc2hpZnQoKTtcblxuICAgIC8vIOeUu+WDj+OBruODnuODvOOCuOODs+OCkuWPluW+l1xuICAgIGNvbnNvbGUuZGVidWcoYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVjdGluZyBpbWFnZSBtYXJnaW5zLi4uYCk7XG4gICAgbGV0IGltYWdlVHJpbW1pbmc6XG4gICAgICB8IHtcbiAgICAgICAgICBtYXJnaW5Ub3A6IG51bWJlcjtcbiAgICAgICAgICBtYXJnaW5Cb3R0b206IG51bWJlcjtcbiAgICAgICAgICBoZWlnaHROZXc6IG51bWJlcjtcbiAgICAgICAgICBoZWlnaHRPbGQ6IG51bWJlcjtcbiAgICAgICAgICB3aWR0aDogbnVtYmVyO1xuICAgICAgICB9XG4gICAgICB8IHVuZGVmaW5lZCA9IHVuZGVmaW5lZDtcbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgbGV0IGltYWdlVHJpbW1lciA9IG5ldyBJbWFnZVRyaW1tZXIoKTtcbiAgICAgIGlmICghcG9zZS5mcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UuZnJhbWVJbWFnZURhdGFVcmwpO1xuXG4gICAgICBjb25zdCBtYXJnaW5Db2xvciA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXRNYXJnaW5Db2xvcigpO1xuICAgICAgY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVjdGVkIG1hcmdpbiBjb2xvci4uLmAsXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzLFxuICAgICAgICBtYXJnaW5Db2xvclxuICAgICAgKTtcbiAgICAgIGlmIChtYXJnaW5Db2xvciA9PT0gbnVsbCkgY29udGludWU7XG4gICAgICBpZiAobWFyZ2luQ29sb3IgIT09IHRoaXMuSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0NPTE9SKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgY29uc3QgdHJpbW1lZCA9IGF3YWl0IGltYWdlVHJpbW1lci50cmltTWFyZ2luKFxuICAgICAgICBtYXJnaW5Db2xvcixcbiAgICAgICAgdGhpcy5JTUFHRV9NQVJHSU5fVFJJTU1JTkdfRElGRl9USFJFU0hPTERcbiAgICAgICk7XG4gICAgICBpZiAoIXRyaW1tZWQpIGNvbnRpbnVlO1xuICAgICAgaW1hZ2VUcmltbWluZyA9IHRyaW1tZWQ7XG4gICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gRGV0ZXJtaW5lZCBpbWFnZSB0cmltbWluZyBwb3NpdGlvbnMuLi5gLFxuICAgICAgICB0cmltbWVkXG4gICAgICApO1xuICAgICAgYnJlYWs7XG4gICAgfVxuXG4gICAgLy8g55S75YOP44KS5pW05b2iXG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGxldCBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwgfHwgIXBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cblxuICAgICAgY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIFByb2Nlc3NpbmcgaW1hZ2UuLi5gLFxuICAgICAgICBwb3NlLnRpbWVNaWxpc2Vjb25kc1xuICAgICAgKTtcblxuICAgICAgLy8g55S75YOP44KS5pW05b2iIC0g44OV44Os44O844Og55S75YOPXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgaWYgKGltYWdlVHJpbW1pbmcpIHtcbiAgICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmNyb3AoXG4gICAgICAgICAgMCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLm1hcmdpblRvcCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLndpZHRoLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcuaGVpZ2h0TmV3XG4gICAgICAgICk7XG4gICAgICB9XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXBsYWNlQ29sb3IoXG4gICAgICAgIHRoaXMuSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX1NSQ19DT0xPUixcbiAgICAgICAgdGhpcy5JTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRFNUX0NPTE9SLFxuICAgICAgICB0aGlzLklNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9ESUZGX1RIUkVTSE9MRFxuICAgICAgKTtcblxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLnJlc2l6ZVdpdGhGaXQoe1xuICAgICAgICB3aWR0aDogdGhpcy5JTUFHRV9XSURUSCxcbiAgICAgIH0pO1xuXG4gICAgICBsZXQgbmV3RGF0YVVybCA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXREYXRhVXJsKFxuICAgICAgICB0aGlzLklNQUdFX01JTUUsXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL2pwZWcnIHx8IHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL3dlYnAnXG4gICAgICAgICAgPyB0aGlzLklNQUdFX1FVQUxJVFlcbiAgICAgICAgICA6IHVuZGVmaW5lZFxuICAgICAgKTtcbiAgICAgIGlmICghbmV3RGF0YVVybCkge1xuICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIENvdWxkIG5vdCBnZXQgdGhlIG5ldyBkYXRhdXJsIGZvciBmcmFtZSBpbWFnZWBcbiAgICAgICAgKTtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsID0gbmV3RGF0YVVybDtcblxuICAgICAgLy8g55S75YOP44KS5pW05b2iIC0g44Od44O844K644OX44Os44OT44Ol44O855S75YOPXG4gICAgICBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLnBvc2VJbWFnZURhdGFVcmwpO1xuXG4gICAgICBpZiAoaW1hZ2VUcmltbWluZykge1xuICAgICAgICBhd2FpdCBpbWFnZVRyaW1tZXIuY3JvcChcbiAgICAgICAgICAwLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcubWFyZ2luVG9wLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcud2lkdGgsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5oZWlnaHROZXdcbiAgICAgICAgKTtcbiAgICAgIH1cblxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLnJlc2l6ZVdpdGhGaXQoe1xuICAgICAgICB3aWR0aDogdGhpcy5JTUFHRV9XSURUSCxcbiAgICAgIH0pO1xuXG4gICAgICBuZXdEYXRhVXJsID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldERhdGFVcmwoXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSxcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2UvanBlZycgfHwgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2Uvd2VicCdcbiAgICAgICAgICA/IHRoaXMuSU1BR0VfUVVBTElUWVxuICAgICAgICAgIDogdW5kZWZpbmVkXG4gICAgICApO1xuICAgICAgaWYgKCFuZXdEYXRhVXJsKSB7XG4gICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gQ291bGQgbm90IGdldCB0aGUgbmV3IGRhdGF1cmwgZm9yIHBvc2UgcHJldmlldyBpbWFnZWBcbiAgICAgICAgKTtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBwb3NlLnBvc2VJbWFnZURhdGFVcmwgPSBuZXdEYXRhVXJsO1xuXG4gICAgICBpZiAocG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgLy8g55S75YOP44KS5pW05b2iIC0g6aGU44OV44Os44O844Og55S75YOPXG4gICAgICAgIGltYWdlVHJpbW1lciA9IG5ldyBJbWFnZVRyaW1tZXIoKTtcbiAgICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwpO1xuXG4gICAgICAgIG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgICB0aGlzLklNQUdFX01JTUUsXG4gICAgICAgICAgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2UvanBlZycgfHwgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2Uvd2VicCdcbiAgICAgICAgICAgID8gdGhpcy5JTUFHRV9RVUFMSVRZXG4gICAgICAgICAgICA6IHVuZGVmaW5lZFxuICAgICAgICApO1xuICAgICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gQ291bGQgbm90IGdldCB0aGUgbmV3IGRhdGF1cmwgZm9yIGZhY2UgZnJhbWUgaW1hZ2VgXG4gICAgICAgICAgKTtcbiAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgfVxuICAgICAgICBwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdGhpcy5pc0ZpbmFsaXplZCA9IHRydWU7XG4gIH1cblxuICAvKipcbiAgICog6aGe5Ly844Od44O844K644Gu5Y+W5b6XXG4gICAqIEBwYXJhbSByZXN1bHRzIE1lZGlhUGlwZSBIb2xpc3RpYyDjgavjgojjgovjg53jg7zjgrrjga7mpJzlh7rntZDmnpxcbiAgICogQHBhcmFtIHRocmVzaG9sZCDjgZfjgY3jgYTlgKRcbiAgICogQHBhcmFtIHRhcmdldFJhbmdlIOODneODvOOCuuOCkuavlOi8g+OBmeOCi+evhOWbsiAoYWxsOiDlhajjgaYsIGJvZHlQb3NlOiDouqvkvZPjga7jgb8sIGhhbmRQb3NlOiDmiYvmjIfjga7jgb8pXG4gICAqIEByZXR1cm5zIOmhnuS8vOODneODvOOCuuOBrumFjeWIl1xuICAgKi9cbiAgZ2V0U2ltaWxhclBvc2VzKFxuICAgIHJlc3VsdHM6IFJlc3VsdHMsXG4gICAgdGhyZXNob2xkOiBudW1iZXIgPSAwLjksXG4gICAgdGFyZ2V0UmFuZ2U6ICdhbGwnIHwgJ2JvZHlQb3NlJyB8ICdoYW5kUG9zZScgPSAnYWxsJ1xuICApOiBTaW1pbGFyUG9zZUl0ZW1bXSB7XG4gICAgLy8g6Lqr5L2T44Gu44OZ44Kv44OI44Or44KS5Y+W5b6XXG4gICAgbGV0IGJvZHlWZWN0b3I6IEJvZHlWZWN0b3I7XG4gICAgdHJ5IHtcbiAgICAgIGJvZHlWZWN0b3IgPSBQb3NlU2V0LmdldEJvZHlWZWN0b3IoKHJlc3VsdHMgYXMgYW55KS5lYSk7XG4gICAgfSBjYXRjaCAoZSkge1xuICAgICAgY29uc29sZS5lcnJvcihgW1Bvc2VTZXRdIGdldFNpbWlsYXJQb3NlcyAtIEVycm9yIG9jY3VycmVkYCwgZSwgcmVzdWx0cyk7XG4gICAgICByZXR1cm4gW107XG4gICAgfVxuICAgIGlmICghYm9keVZlY3Rvcikge1xuICAgICAgdGhyb3cgJ0NvdWxkIG5vdCBnZXQgdGhlIGJvZHkgdmVjdG9yJztcbiAgICB9XG5cbiAgICAvLyDmiYvmjIfjga7jg5njgq/jg4jjg6vjgpLlj5blvpdcbiAgICBsZXQgaGFuZFZlY3RvcjogSGFuZFZlY3RvcjtcbiAgICBpZiAodGFyZ2V0UmFuZ2UgPT09ICdhbGwnIHx8IHRhcmdldFJhbmdlID09PSAnaGFuZFBvc2UnKSB7XG4gICAgICBoYW5kVmVjdG9yID0gUG9zZVNldC5nZXRIYW5kVmVjdG9yKFxuICAgICAgICByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzLFxuICAgICAgICByZXN1bHRzLnJpZ2h0SGFuZExhbmRtYXJrc1xuICAgICAgKTtcbiAgICAgIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2hhbmRQb3NlJyAmJiAhaGFuZFZlY3Rvcikge1xuICAgICAgICB0aHJvdyAnQ291bGQgbm90IGdldCB0aGUgaGFuZCB2ZWN0b3InO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIOWQhOODneODvOOCuuOBqOODmeOCr+ODiOODq+OCkuavlOi8g1xuICAgIGNvbnN0IHBvc2VzID0gW107XG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGlmIChcbiAgICAgICAgKHRhcmdldFJhbmdlID09PSAnYWxsJyB8fCB0YXJnZXRSYW5nZSA9PT0gJ2JvZHlQb3NlJykgJiZcbiAgICAgICAgIXBvc2UuYm9keVZlY3RvclxuICAgICAgKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfSBlbHNlIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2hhbmRQb3NlJyAmJiAhcG9zZS5oYW5kVmVjdG9yKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuXG4gICAgICAvKmNvbnNvbGUuZGVidWcoXG4gICAgICAgICdbUG9zZVNldF0gZ2V0U2ltaWxhclBvc2VzIC0gJyxcbiAgICAgICAgdGhpcy5nZXRWaWRlb05hbWUoKSxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHNcbiAgICAgICk7Ki9cblxuICAgICAgLy8g6Lqr5L2T44Gu44Od44O844K644Gu6aGe5Ly85bqm44KS5Y+W5b6XXG4gICAgICBsZXQgYm9keVNpbWlsYXJpdHk6IG51bWJlcjtcbiAgICAgIGlmIChib2R5VmVjdG9yICYmIHBvc2UuYm9keVZlY3Rvcikge1xuICAgICAgICBib2R5U2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0Qm9keVBvc2VTaW1pbGFyaXR5KFxuICAgICAgICAgIHBvc2UuYm9keVZlY3RvcixcbiAgICAgICAgICBib2R5VmVjdG9yXG4gICAgICAgICk7XG4gICAgICB9XG5cbiAgICAgIC8vIOaJi+aMh+OBruODneODvOOCuuOBrumhnuS8vOW6puOCkuWPluW+l1xuICAgICAgbGV0IGhhbmRTaW1pbGFyaXR5OiBudW1iZXI7XG4gICAgICBpZiAoaGFuZFZlY3RvciAmJiBwb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgaGFuZFNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEhhbmRTaW1pbGFyaXR5KHBvc2UuaGFuZFZlY3RvciwgaGFuZFZlY3Rvcik7XG4gICAgICB9XG5cbiAgICAgIC8vIOWIpOWumlxuICAgICAgbGV0IHNpbWlsYXJpdHk6IG51bWJlcixcbiAgICAgICAgaXNTaW1pbGFyID0gZmFsc2U7XG4gICAgICBpZiAodGFyZ2V0UmFuZ2UgPT09ICdhbGwnKSB7XG4gICAgICAgIHNpbWlsYXJpdHkgPSBNYXRoLm1heChib2R5U2ltaWxhcml0eSA/PyAwLCBoYW5kU2ltaWxhcml0eSA/PyAwKTtcbiAgICAgICAgaWYgKHRocmVzaG9sZCA8PSBib2R5U2ltaWxhcml0eSB8fCB0aHJlc2hvbGQgPD0gaGFuZFNpbWlsYXJpdHkpIHtcbiAgICAgICAgICBpc1NpbWlsYXIgPSB0cnVlO1xuICAgICAgICB9XG4gICAgICB9IGVsc2UgaWYgKHRhcmdldFJhbmdlID09PSAnYm9keVBvc2UnKSB7XG4gICAgICAgIHNpbWlsYXJpdHkgPSBib2R5U2ltaWxhcml0eTtcbiAgICAgICAgaWYgKHRocmVzaG9sZCA8PSBib2R5U2ltaWxhcml0eSkge1xuICAgICAgICAgIGlzU2ltaWxhciA9IHRydWU7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSBpZiAodGFyZ2V0UmFuZ2UgPT09ICdoYW5kUG9zZScpIHtcbiAgICAgICAgc2ltaWxhcml0eSA9IGhhbmRTaW1pbGFyaXR5O1xuICAgICAgICBpZiAodGhyZXNob2xkIDw9IGhhbmRTaW1pbGFyaXR5KSB7XG4gICAgICAgICAgaXNTaW1pbGFyID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICBpZiAoIWlzU2ltaWxhcikgY29udGludWU7XG5cbiAgICAgIC8vIOe1kOaenOOBuOi/veWKoFxuICAgICAgcG9zZXMucHVzaCh7XG4gICAgICAgIC4uLnBvc2UsXG4gICAgICAgIHNpbWlsYXJpdHk6IHNpbWlsYXJpdHksXG4gICAgICAgIGJvZHlQb3NlU2ltaWxhcml0eTogYm9keVNpbWlsYXJpdHksXG4gICAgICAgIGhhbmRQb3NlU2ltaWxhcml0eTogaGFuZFNpbWlsYXJpdHksXG4gICAgICB9IGFzIFNpbWlsYXJQb3NlSXRlbSk7XG4gICAgfVxuXG4gICAgcmV0dXJuIHBvc2VzO1xuICB9XG5cbiAgLyoqXG4gICAqIOi6q+S9k+OBruWnv+WLouOCkuihqOOBmeODmeOCr+ODiOODq+OBruWPluW+l1xuICAgKiBAcGFyYW0gcG9zZUxhbmRtYXJrcyBNZWRpYVBpcGUgSG9saXN0aWMg44Gn5Y+W5b6X44Gn44GN44Gf6Lqr5L2T44Gu44Ov44O844Or44OJ5bqn5qiZIChyYSDphY3liJcpXG4gICAqIEByZXR1cm5zIOODmeOCr+ODiOODq1xuICAgKi9cbiAgc3RhdGljIGdldEJvZHlWZWN0b3IoXG4gICAgcG9zZUxhbmRtYXJrczogeyB4OiBudW1iZXI7IHk6IG51bWJlcjsgejogbnVtYmVyIH1bXVxuICApOiBCb2R5VmVjdG9yIHwgdW5kZWZpbmVkIHtcbiAgICByZXR1cm4ge1xuICAgICAgcmlnaHRXcmlzdFRvUmlnaHRFbGJvdzogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1dSSVNUXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1dSSVNUXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1dSSVNUXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS56LFxuICAgICAgXSxcbiAgICAgIHJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXI6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9TSE9VTERFUl0ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9TSE9VTERFUl0ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9TSE9VTERFUl0ueixcbiAgICAgIF0sXG4gICAgICBsZWZ0V3Jpc3RUb0xlZnRFbGJvdzogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS56LFxuICAgICAgXSxcbiAgICAgIGxlZnRFbGJvd1RvTGVmdFNob3VsZGVyOiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLnosXG4gICAgICBdLFxuICAgIH07XG4gIH1cblxuICAvKipcbiAgICog5omL5oyH44Gu5ae/5Yui44KS6KGo44GZ44OZ44Kv44OI44Or44Gu5Y+W5b6XXG4gICAqIEBwYXJhbSBsZWZ0SGFuZExhbmRtYXJrcyBNZWRpYVBpcGUgSG9saXN0aWMg44Gn5Y+W5b6X44Gn44GN44Gf5bem5omL44Gu5q2j6KaP5YyW5bqn5qiZXG4gICAqIEBwYXJhbSByaWdodEhhbmRMYW5kbWFya3MgTWVkaWFQaXBlIEhvbGlzdGljIOOBp+WPluW+l+OBp+OBjeOBn+WPs+aJi+OBruato+imj+WMluW6p+aomVxuICAgKiBAcmV0dXJucyDjg5njgq/jg4jjg6tcbiAgICovXG4gIHN0YXRpYyBnZXRIYW5kVmVjdG9yKFxuICAgIGxlZnRIYW5kTGFuZG1hcmtzOiB7IHg6IG51bWJlcjsgeTogbnVtYmVyOyB6OiBudW1iZXIgfVtdLFxuICAgIHJpZ2h0SGFuZExhbmRtYXJrczogeyB4OiBudW1iZXI7IHk6IG51bWJlcjsgejogbnVtYmVyIH1bXVxuICApOiBIYW5kVmVjdG9yIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAoXG4gICAgICAocmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMCkgJiZcbiAgICAgIChsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMClcbiAgICApIHtcbiAgICAgIHJldHVybiB1bmRlZmluZWQ7XG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgIC8vIOWPs+aJiyAtIOimquaMh1xuICAgICAgcmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbNF0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1szXS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbNF0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1szXS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbNF0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1szXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIHJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbM10ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1syXS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbM10ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1syXS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbM10ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1syXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOS6uuW3ruOBl+aMh1xuICAgICAgcmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbOF0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbOF0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbOF0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIHJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbN10ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1s2XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbN10ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1s2XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbN10ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1s2XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOS4reaMh1xuICAgICAgcmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzEyXS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzExXS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTJdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTFdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMl0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMF0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzExXS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzEwXS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTFdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTBdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5Y+z5omLIC0g6Jas5oyHXG4gICAgICByaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE2XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTZdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTVdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNl0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTVdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTRdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNF0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE0XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICAgcmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMjBdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTldLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1syMF0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzIwXS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIHJpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTldLnggLSByaWdodEhhbmRMYW5kbWFya3NbMThdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOF0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE4XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOimquaMh1xuICAgICAgbGVmdFRodW1iVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s0XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbM10ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbNF0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzRdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1szXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIGxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1szXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMl0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbM10ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzJdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1syXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOS6uuW3ruOBl+aMh1xuICAgICAgbGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s4XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbN10ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbOF0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzddLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzhdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1s3XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIGxlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s3XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbNl0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbN10ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzZdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzddLnogLSBsZWZ0SGFuZExhbmRtYXJrc1s2XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOS4reaMh1xuICAgICAgbGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTJdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTJdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTJdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzExXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTBdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzExXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTBdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzExXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTBdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g6Jas5oyHXG4gICAgICBsZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTZdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTZdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTZdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzE0XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzE0XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzE0XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOWwj+aMh1xuICAgICAgbGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1syMF0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1syMF0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1syMF0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIGxlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzE4XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzE4XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzE4XS56LFxuICAgICAgICAgICAgXSxcbiAgICB9O1xuICB9XG5cbiAgLyoqXG4gICAqIEJvZHlWZWN0b3Ig6ZaT44GM6aGe5Ly844GX44Gm44GE44KL44GL44Gp44GG44GL44Gu5Yik5a6aXG4gICAqIEBwYXJhbSBib2R5VmVjdG9yQSDmr5TovIPlhYjjga4gQm9keVZlY3RvclxuICAgKiBAcGFyYW0gYm9keVZlY3RvckIg5q+U6LyD5YWD44GuIEJvZHlWZWN0b3JcbiAgICogQHBhcmFtIHRocmVzaG9sZCDjgZfjgY3jgYTlgKRcbiAgICogQHJldHVybnMg6aGe5Ly844GX44Gm44GE44KL44GL44Gp44GG44GLXG4gICAqL1xuICBzdGF0aWMgaXNTaW1pbGFyQm9keVBvc2UoXG4gICAgYm9keVZlY3RvckE6IEJvZHlWZWN0b3IsXG4gICAgYm9keVZlY3RvckI6IEJvZHlWZWN0b3IsXG4gICAgdGhyZXNob2xkID0gMC44XG4gICk6IGJvb2xlYW4ge1xuICAgIGxldCBpc1NpbWlsYXIgPSBmYWxzZTtcbiAgICBjb25zdCBzaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRCb2R5UG9zZVNpbWlsYXJpdHkoYm9keVZlY3RvckEsIGJvZHlWZWN0b3JCKTtcbiAgICBpZiAoc2ltaWxhcml0eSA+PSB0aHJlc2hvbGQpIGlzU2ltaWxhciA9IHRydWU7XG5cbiAgICAvLyBjb25zb2xlLmRlYnVnKGBbUG9zZVNldF0gaXNTaW1pbGFyUG9zZWAsIGlzU2ltaWxhciwgc2ltaWxhcml0eSk7XG5cbiAgICByZXR1cm4gaXNTaW1pbGFyO1xuICB9XG5cbiAgLyoqXG4gICAqIOi6q+S9k+ODneODvOOCuuOBrumhnuS8vOW6puOBruWPluW+l1xuICAgKiBAcGFyYW0gYm9keVZlY3RvckEg5q+U6LyD5YWI44GuIEJvZHlWZWN0b3JcbiAgICogQHBhcmFtIGJvZHlWZWN0b3JCIOavlOi8g+WFg+OBriBCb2R5VmVjdG9yXG4gICAqIEByZXR1cm5zIOmhnuS8vOW6plxuICAgKi9cbiAgc3RhdGljIGdldEJvZHlQb3NlU2ltaWxhcml0eShcbiAgICBib2R5VmVjdG9yQTogQm9keVZlY3RvcixcbiAgICBib2R5VmVjdG9yQjogQm9keVZlY3RvclxuICApOiBudW1iZXIge1xuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllcyA9IHtcbiAgICAgIGxlZnRXcmlzdFRvTGVmdEVsYm93OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLmxlZnRXcmlzdFRvTGVmdEVsYm93LFxuICAgICAgICBib2R5VmVjdG9yQi5sZWZ0V3Jpc3RUb0xlZnRFbGJvd1xuICAgICAgKSxcbiAgICAgIGxlZnRFbGJvd1RvTGVmdFNob3VsZGVyOiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLmxlZnRFbGJvd1RvTGVmdFNob3VsZGVyLFxuICAgICAgICBib2R5VmVjdG9yQi5sZWZ0RWxib3dUb0xlZnRTaG91bGRlclxuICAgICAgKSxcbiAgICAgIHJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3c6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgYm9keVZlY3RvckEucmlnaHRXcmlzdFRvUmlnaHRFbGJvdyxcbiAgICAgICAgYm9keVZlY3RvckIucmlnaHRXcmlzdFRvUmlnaHRFbGJvd1xuICAgICAgKSxcbiAgICAgIHJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXI6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgYm9keVZlY3RvckEucmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcixcbiAgICAgICAgYm9keVZlY3RvckIucmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlclxuICAgICAgKSxcbiAgICB9O1xuXG4gICAgY29uc3QgY29zU2ltaWxhcml0aWVzU3VtID0gT2JqZWN0LnZhbHVlcyhjb3NTaW1pbGFyaXRpZXMpLnJlZHVjZShcbiAgICAgIChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSxcbiAgICAgIDBcbiAgICApO1xuICAgIHJldHVybiBjb3NTaW1pbGFyaXRpZXNTdW0gLyBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXMpLmxlbmd0aDtcbiAgfVxuXG4gIC8qKlxuICAgKiBIYW5kVmVjdG9yIOmWk+OBjOmhnuS8vOOBl+OBpuOBhOOCi+OBi+OBqeOBhuOBi+OBruWIpOWumlxuICAgKiBAcGFyYW0gaGFuZFZlY3RvckEg5q+U6LyD5YWI44GuIEhhbmRWZWN0b3JcbiAgICogQHBhcmFtIGhhbmRWZWN0b3JCIOavlOi8g+WFg+OBriBIYW5kVmVjdG9yXG4gICAqIEBwYXJhbSB0aHJlc2hvbGQg44GX44GN44GE5YCkXG4gICAqIEByZXR1cm5zIOmhnuS8vOOBl+OBpuOBhOOCi+OBi+OBqeOBhuOBi1xuICAgKi9cbiAgc3RhdGljIGlzU2ltaWxhckhhbmRQb3NlKFxuICAgIGhhbmRWZWN0b3JBOiBIYW5kVmVjdG9yLFxuICAgIGhhbmRWZWN0b3JCOiBIYW5kVmVjdG9yLFxuICAgIHRocmVzaG9sZCA9IDAuNzVcbiAgKTogYm9vbGVhbiB7XG4gICAgY29uc3Qgc2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0SGFuZFNpbWlsYXJpdHkoaGFuZFZlY3RvckEsIGhhbmRWZWN0b3JCKTtcbiAgICBpZiAoc2ltaWxhcml0eSA9PT0gLTEpIHtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cbiAgICByZXR1cm4gc2ltaWxhcml0eSA+PSB0aHJlc2hvbGQ7XG4gIH1cblxuICAvKipcbiAgICog5omL44Gu44Od44O844K644Gu6aGe5Ly85bqm44Gu5Y+W5b6XXG4gICAqIEBwYXJhbSBoYW5kVmVjdG9yQSDmr5TovIPlhYjjga4gSGFuZFZlY3RvclxuICAgKiBAcGFyYW0gaGFuZFZlY3RvckIg5q+U6LyD5YWD44GuIEhhbmRWZWN0b3JcbiAgICogQHJldHVybnMg6aGe5Ly85bqmXG4gICAqL1xuICBzdGF0aWMgZ2V0SGFuZFNpbWlsYXJpdHkoXG4gICAgaGFuZFZlY3RvckE6IEhhbmRWZWN0b3IsXG4gICAgaGFuZFZlY3RvckI6IEhhbmRWZWN0b3JcbiAgKTogbnVtYmVyIHtcbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQgPVxuICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ID09PSBudWxsIHx8XG4gICAgICBoYW5kVmVjdG9yQi5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICAgPyB1bmRlZmluZWRcbiAgICAgICAgOiB7XG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDopqrmjIdcbiAgICAgICAgICAgIHJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOS6uuW3ruOBl+aMh1xuICAgICAgICAgICAgcmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5Y+z5omLIC0g5Lit5oyHXG4gICAgICAgICAgICByaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDolqzmjIdcbiAgICAgICAgICAgIHJpZ2h0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5Y+z5omLIC0g5bCP5oyHXG4gICAgICAgICAgICByaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgfTtcblxuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kID1cbiAgICAgIGhhbmRWZWN0b3JBLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ID09PSBudWxsIHx8XG4gICAgICBoYW5kVmVjdG9yQi5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbFxuICAgICAgICA/IHVuZGVmaW5lZFxuICAgICAgICA6IHtcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOimquaMh1xuICAgICAgICAgICAgbGVmdFRodW1iVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFRodW1iVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5Lq65beu44GX5oyHXG4gICAgICAgICAgICBsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDkuK3mjIdcbiAgICAgICAgICAgIGxlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIGxlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOiWrOaMh1xuICAgICAgICAgICAgbGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5bCP5oyHXG4gICAgICAgICAgICBsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgfTtcblxuICAgIC8vIOW3puaJi+OBrumhnuS8vOW6plxuICAgIGxldCBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCA9IDA7XG4gICAgaWYgKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kKSB7XG4gICAgICBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCA9IE9iamVjdC52YWx1ZXMoXG4gICAgICAgIGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kXG4gICAgICApLnJlZHVjZSgoc3VtLCB2YWx1ZSkgPT4gc3VtICsgdmFsdWUsIDApO1xuICAgIH1cblxuICAgIC8vIOWPs+aJi+OBrumhnuS8vOW6plxuICAgIGxldCBjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgPSAwO1xuICAgIGlmIChjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQpIHtcbiAgICAgIGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCA9IE9iamVjdC52YWx1ZXMoXG4gICAgICAgIGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZFxuICAgICAgKS5yZWR1Y2UoKHN1bSwgdmFsdWUpID0+IHN1bSArIHZhbHVlLCAwKTtcbiAgICB9XG5cbiAgICAvLyDlkIjnrpfjgZXjgozjgZ/poZ7kvLzluqZcbiAgICBpZiAoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kICYmIGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kKSB7XG4gICAgICByZXR1cm4gKFxuICAgICAgICAoY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kICsgY29zU2ltaWxhcml0aWVzU3VtTGVmdEhhbmQpIC9cbiAgICAgICAgKE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCEpLmxlbmd0aCArXG4gICAgICAgICAgT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzTGVmdEhhbmQhKS5sZW5ndGgpXG4gICAgICApO1xuICAgIH0gZWxzZSBpZiAoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kKSB7XG4gICAgICBpZiAoXG4gICAgICAgIGhhbmRWZWN0b3JCLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ICE9PSBudWxsICYmXG4gICAgICAgIGhhbmRWZWN0b3JBLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ID09PSBudWxsXG4gICAgICApIHtcbiAgICAgICAgLy8gaGFuZFZlY3RvckIg44Gn5bem5omL44GM44GC44KL44Gu44GrIGhhbmRWZWN0b3JBIOOBp+W3puaJi+OBjOOBquOBhOWgtOWQiOOAgemhnuS8vOW6puOCkua4m+OCieOBmVxuICAgICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICAgIGBbUG9zZVNldF0gZ2V0SGFuZFNpbWlsYXJpdHkgLSBBZGp1c3Qgc2ltaWxhcml0eSwgYmVjYXVzZSBsZWZ0IGhhbmQgbm90IGZvdW5kLi4uYFxuICAgICAgICApO1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgIGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCAvXG4gICAgICAgICAgKE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCEpLmxlbmd0aCAqIDIpXG4gICAgICAgICk7XG4gICAgICB9XG4gICAgICByZXR1cm4gKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgL1xuICAgICAgICBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQhKS5sZW5ndGhcbiAgICAgICk7XG4gICAgfSBlbHNlIGlmIChjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCkge1xuICAgICAgaWYgKFxuICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgIT09IG51bGwgJiZcbiAgICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ID09PSBudWxsXG4gICAgICApIHtcbiAgICAgICAgLy8gaGFuZFZlY3RvckIg44Gn5Y+z5omL44GM44GC44KL44Gu44GrIGhhbmRWZWN0b3JBIOOBp+WPs+aJi+OBjOOBquOBhOWgtOWQiOOAgemhnuS8vOW6puOCkua4m+OCieOBmVxuICAgICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICAgIGBbUG9zZVNldF0gZ2V0SGFuZFNpbWlsYXJpdHkgLSBBZGp1c3Qgc2ltaWxhcml0eSwgYmVjYXVzZSByaWdodCBoYW5kIG5vdCBmb3VuZC4uLmBcbiAgICAgICAgKTtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCAvXG4gICAgICAgICAgKE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kISkubGVuZ3RoICogMilcbiAgICAgICAgKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiAoXG4gICAgICAgIGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kIC9cbiAgICAgICAgT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzTGVmdEhhbmQhKS5sZW5ndGhcbiAgICAgICk7XG4gICAgfVxuXG4gICAgcmV0dXJuIC0xO1xuICB9XG5cbiAgLyoqXG4gICAqIFpJUCDjg5XjgqHjgqTjg6vjgajjgZfjgabjga7jgrfjg6rjgqLjg6njgqTjgrpcbiAgICogQHJldHVybnMgWklQ44OV44Kh44Kk44OrIChCbG9iIOW9ouW8jylcbiAgICovXG4gIHB1YmxpYyBhc3luYyBnZXRaaXAoKTogUHJvbWlzZTxCbG9iPiB7XG4gICAgY29uc3QganNaaXAgPSBuZXcgSlNaaXAoKTtcbiAgICBqc1ppcC5maWxlKCdwb3Nlcy5qc29uJywgYXdhaXQgdGhpcy5nZXRKc29uKCkpO1xuXG4gICAgY29uc3QgaW1hZ2VGaWxlRXh0ID0gdGhpcy5nZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKHRoaXMuSU1BR0VfTUlNRSk7XG5cbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgaWYgKHBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICBjb25zdCBpbmRleCA9XG4gICAgICAgICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5mcmFtZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYGZyYW1lLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7aW1hZ2VGaWxlRXh0fWAsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZyYW1lIGltYWdlYCxcbiAgICAgICAgICAgIGVycm9yXG4gICAgICAgICAgKTtcbiAgICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgaWYgKHBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UucG9zZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYHBvc2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtpbWFnZUZpbGVFeHR9YCwgYmFzZTY0LCB7XG4gICAgICAgICAgICBiYXNlNjQ6IHRydWUsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlRXhwb3J0ZXJTZXJ2aWNlXSBwdXNoIC0gQ291bGQgbm90IHB1c2ggZnJhbWUgaW1hZ2VgLFxuICAgICAgICAgICAgZXJyb3JcbiAgICAgICAgICApO1xuICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBpZiAocG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICBjb25zdCBpbmRleCA9XG4gICAgICAgICAgICBwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsLnN1YnN0cmluZyhpbmRleCk7XG4gICAgICAgICAganNaaXAuZmlsZShgZmFjZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ltYWdlRmlsZUV4dH1gLCBiYXNlNjQsIHtcbiAgICAgICAgICAgIGJhc2U2NDogdHJ1ZSxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgW1Bvc2VFeHBvcnRlclNlcnZpY2VdIHB1c2ggLSBDb3VsZCBub3QgcHVzaCBmYWNlIGZyYW1lIGltYWdlYCxcbiAgICAgICAgICAgIGVycm9yXG4gICAgICAgICAgKTtcbiAgICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiBhd2FpdCBqc1ppcC5nZW5lcmF0ZUFzeW5jKHsgdHlwZTogJ2Jsb2InIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIEpTT04g5paH5a2X5YiX44Go44GX44Gm44Gu44K344Oq44Ki44Op44Kk44K6XG4gICAqIEByZXR1cm5zIEpTT04g5paH5a2X5YiXXG4gICAqL1xuICBwdWJsaWMgYXN5bmMgZ2V0SnNvbigpOiBQcm9taXNlPHN0cmluZz4ge1xuICAgIGlmICh0aGlzLnZpZGVvTWV0YWRhdGEgPT09IHVuZGVmaW5lZCB8fCB0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpXG4gICAgICByZXR1cm4gJ3t9JztcblxuICAgIGlmICghdGhpcy5pc0ZpbmFsaXplZCkge1xuICAgICAgYXdhaXQgdGhpcy5maW5hbGl6ZSgpO1xuICAgIH1cblxuICAgIGxldCBwb3NlTGFuZG1hcmtNYXBwaW5ncyA9IFtdO1xuICAgIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKFBPU0VfTEFORE1BUktTKSkge1xuICAgICAgY29uc3QgaW5kZXg6IG51bWJlciA9IFBPU0VfTEFORE1BUktTW2tleSBhcyBrZXlvZiB0eXBlb2YgUE9TRV9MQU5ETUFSS1NdO1xuICAgICAgcG9zZUxhbmRtYXJrTWFwcGluZ3NbaW5kZXhdID0ga2V5O1xuICAgIH1cblxuICAgIGNvbnN0IGpzb246IFBvc2VTZXRKc29uID0ge1xuICAgICAgZ2VuZXJhdG9yOiAnbXAtdmlkZW8tcG9zZS1leHRyYWN0b3InLFxuICAgICAgdmVyc2lvbjogMSxcbiAgICAgIHZpZGVvOiB0aGlzLnZpZGVvTWV0YWRhdGEhLFxuICAgICAgcG9zZXM6IHRoaXMucG9zZXMubWFwKChwb3NlOiBQb3NlU2V0SXRlbSk6IFBvc2VTZXRKc29uSXRlbSA9PiB7XG4gICAgICAgIC8vIEJvZHlWZWN0b3Ig44Gu5Zyn57iuXG4gICAgICAgIGNvbnN0IGJvZHlWZWN0b3IgPSBbXTtcbiAgICAgICAgZm9yIChjb25zdCBrZXkgb2YgUG9zZVNldC5CT0RZX1ZFQ1RPUl9NQVBQSU5HUykge1xuICAgICAgICAgIGJvZHlWZWN0b3IucHVzaChwb3NlLmJvZHlWZWN0b3Jba2V5IGFzIGtleW9mIEJvZHlWZWN0b3JdKTtcbiAgICAgICAgfVxuXG4gICAgICAgIC8vIEhhbmRWZWN0b3Ig44Gu5Zyn57iuXG4gICAgICAgIGxldCBoYW5kVmVjdG9yOiAobnVtYmVyW10gfCBudWxsKVtdIHwgdW5kZWZpbmVkID0gdW5kZWZpbmVkO1xuICAgICAgICBpZiAocG9zZS5oYW5kVmVjdG9yKSB7XG4gICAgICAgICAgaGFuZFZlY3RvciA9IFtdO1xuICAgICAgICAgIGZvciAoY29uc3Qga2V5IG9mIFBvc2VTZXQuSEFORF9WRUNUT1JfTUFQUElOR1MpIHtcbiAgICAgICAgICAgIGhhbmRWZWN0b3IucHVzaChwb3NlLmhhbmRWZWN0b3Jba2V5IGFzIGtleW9mIEhhbmRWZWN0b3JdKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cblxuICAgICAgICAvLyBQb3NlU2V0SnNvbkl0ZW0g44GuIHBvc2Ug44Kq44OW44K444Kn44Kv44OI44KS55Sf5oiQXG4gICAgICAgIHJldHVybiB7XG4gICAgICAgICAgdDogcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgICAgZDogcG9zZS5kdXJhdGlvbk1pbGlzZWNvbmRzLFxuICAgICAgICAgIHA6IHBvc2UucG9zZSxcbiAgICAgICAgICBsOiBwb3NlLmxlZnRIYW5kLFxuICAgICAgICAgIHI6IHBvc2UucmlnaHRIYW5kLFxuICAgICAgICAgIHY6IGJvZHlWZWN0b3IsXG4gICAgICAgICAgaDogaGFuZFZlY3RvcixcbiAgICAgICAgICBlOiBwb3NlLmV4dGVuZGVkRGF0YSxcbiAgICAgICAgICBtZDogcG9zZS5tZXJnZWREdXJhdGlvbk1pbGlzZWNvbmRzLFxuICAgICAgICAgIG10OiBwb3NlLm1lcmdlZFRpbWVNaWxpc2Vjb25kcyxcbiAgICAgICAgfTtcbiAgICAgIH0pLFxuICAgICAgcG9zZUxhbmRtYXJrTWFwcHBpbmdzOiBwb3NlTGFuZG1hcmtNYXBwaW5ncyxcbiAgICB9O1xuXG4gICAgcmV0dXJuIEpTT04uc3RyaW5naWZ5KGpzb24pO1xuICB9XG5cbiAgLyoqXG4gICAqIEpTT04g44GL44KJ44Gu6Kqt44G/6L6844G/XG4gICAqIEBwYXJhbSBqc29uIEpTT04g5paH5a2X5YiXIOOBvuOBn+OBryBKU09OIOOCquODluOCuOOCp+OCr+ODiFxuICAgKi9cbiAgbG9hZEpzb24oanNvbjogc3RyaW5nIHwgYW55KSB7XG4gICAgY29uc3QgcGFyc2VkSnNvbiA9IHR5cGVvZiBqc29uID09PSAnc3RyaW5nJyA/IEpTT04ucGFyc2UoanNvbikgOiBqc29uO1xuXG4gICAgaWYgKHBhcnNlZEpzb24uZ2VuZXJhdG9yICE9PSAnbXAtdmlkZW8tcG9zZS1leHRyYWN0b3InKSB7XG4gICAgICB0aHJvdyAn5LiN5q2j44Gq44OV44Kh44Kk44OrJztcbiAgICB9IGVsc2UgaWYgKHBhcnNlZEpzb24udmVyc2lvbiAhPT0gMSkge1xuICAgICAgdGhyb3cgJ+acquWvvuW/nOOBruODkOODvOOCuOODp+ODsyc7XG4gICAgfVxuXG4gICAgdGhpcy52aWRlb01ldGFkYXRhID0gcGFyc2VkSnNvbi52aWRlbztcbiAgICB0aGlzLnBvc2VzID0gcGFyc2VkSnNvbi5wb3Nlcy5tYXAoKGl0ZW06IFBvc2VTZXRKc29uSXRlbSk6IFBvc2VTZXRJdGVtID0+IHtcbiAgICAgIGNvbnN0IGJvZHlWZWN0b3I6IGFueSA9IHt9O1xuICAgICAgUG9zZVNldC5CT0RZX1ZFQ1RPUl9NQVBQSU5HUy5tYXAoKGtleSwgaW5kZXgpID0+IHtcbiAgICAgICAgYm9keVZlY3RvcltrZXkgYXMga2V5b2YgQm9keVZlY3Rvcl0gPSBpdGVtLnZbaW5kZXhdO1xuICAgICAgfSk7XG5cbiAgICAgIGNvbnN0IGhhbmRWZWN0b3I6IGFueSA9IHt9O1xuICAgICAgaWYgKGl0ZW0uaCkge1xuICAgICAgICBQb3NlU2V0LkhBTkRfVkVDVE9SX01BUFBJTkdTLm1hcCgoa2V5LCBpbmRleCkgPT4ge1xuICAgICAgICAgIGhhbmRWZWN0b3Jba2V5IGFzIGtleW9mIEhhbmRWZWN0b3JdID0gaXRlbS5oIVtpbmRleF07XG4gICAgICAgIH0pO1xuICAgICAgfVxuXG4gICAgICByZXR1cm4ge1xuICAgICAgICB0aW1lTWlsaXNlY29uZHM6IGl0ZW0udCxcbiAgICAgICAgZHVyYXRpb25NaWxpc2Vjb25kczogaXRlbS5kLFxuICAgICAgICBwb3NlOiBpdGVtLnAsXG4gICAgICAgIGxlZnRIYW5kOiBpdGVtLmwsXG4gICAgICAgIHJpZ2h0SGFuZDogaXRlbS5yLFxuICAgICAgICBib2R5VmVjdG9yOiBib2R5VmVjdG9yLFxuICAgICAgICBoYW5kVmVjdG9yOiBoYW5kVmVjdG9yLFxuICAgICAgICBmcmFtZUltYWdlRGF0YVVybDogdW5kZWZpbmVkLFxuICAgICAgICBleHRlbmRlZERhdGE6IGl0ZW0uZSxcbiAgICAgICAgZGVidWc6IHVuZGVmaW5lZCxcbiAgICAgICAgbWVyZ2VkRHVyYXRpb25NaWxpc2Vjb25kczogaXRlbS5tZCxcbiAgICAgICAgbWVyZ2VkVGltZU1pbGlzZWNvbmRzOiBpdGVtLm10LFxuICAgICAgfTtcbiAgICB9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBaSVAg44OV44Kh44Kk44Or44GL44KJ44Gu6Kqt44G/6L6844G/XG4gICAqIEBwYXJhbSBidWZmZXIgWklQIOODleOCoeOCpOODq+OBriBCdWZmZXJcbiAgICogQHBhcmFtIGluY2x1ZGVJbWFnZXMg55S75YOP44KS5bGV6ZaL44GZ44KL44GL44Gp44GG44GLXG4gICAqL1xuICBhc3luYyBsb2FkWmlwKGJ1ZmZlcjogQXJyYXlCdWZmZXIsIGluY2x1ZGVJbWFnZXM6IGJvb2xlYW4gPSB0cnVlKSB7XG4gICAgY29uc3QganNaaXAgPSBuZXcgSlNaaXAoKTtcbiAgICBjb25zb2xlLmRlYnVnKGBbUG9zZVNldF0gaW5pdC4uLmApO1xuICAgIGNvbnN0IHppcCA9IGF3YWl0IGpzWmlwLmxvYWRBc3luYyhidWZmZXIsIHsgYmFzZTY0OiBmYWxzZSB9KTtcbiAgICBpZiAoIXppcCkgdGhyb3cgJ1pJUOODleOCoeOCpOODq+OCkuiqreOBv+i+vOOCgeOBvuOBm+OCk+OBp+OBl+OBnyc7XG5cbiAgICBjb25zdCBqc29uID0gYXdhaXQgemlwLmZpbGUoJ3Bvc2VzLmpzb24nKT8uYXN5bmMoJ3RleHQnKTtcbiAgICBpZiAoanNvbiA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICB0aHJvdyAnWklQ44OV44Kh44Kk44Or44GrIHBvc2UuanNvbiDjgYzlkKvjgb7jgozjgabjgYTjgb7jgZvjgpMnO1xuICAgIH1cblxuICAgIHRoaXMubG9hZEpzb24oanNvbik7XG5cbiAgICBjb25zdCBmaWxlRXh0ID0gdGhpcy5nZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKHRoaXMuSU1BR0VfTUlNRSk7XG5cbiAgICBpZiAoaW5jbHVkZUltYWdlcykge1xuICAgICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgICAgY29uc3QgZnJhbWVJbWFnZUZpbGVOYW1lID0gYGZyYW1lLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7ZmlsZUV4dH1gO1xuICAgICAgICAgIGNvbnN0IGltYWdlQmFzZTY0ID0gYXdhaXQgemlwXG4gICAgICAgICAgICAuZmlsZShmcmFtZUltYWdlRmlsZU5hbWUpXG4gICAgICAgICAgICA/LmFzeW5jKCdiYXNlNjQnKTtcbiAgICAgICAgICBpZiAoaW1hZ2VCYXNlNjQpIHtcbiAgICAgICAgICAgIHBvc2UuZnJhbWVJbWFnZURhdGFVcmwgPSBgZGF0YToke3RoaXMuSU1BR0VfTUlNRX07YmFzZTY0LCR7aW1hZ2VCYXNlNjR9YDtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgaWYgKCFwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgICBjb25zdCBwb3NlSW1hZ2VGaWxlTmFtZSA9IGBwb3NlLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7ZmlsZUV4dH1gO1xuICAgICAgICAgIGNvbnN0IGltYWdlQmFzZTY0ID0gYXdhaXQgemlwXG4gICAgICAgICAgICAuZmlsZShwb3NlSW1hZ2VGaWxlTmFtZSlcbiAgICAgICAgICAgID8uYXN5bmMoJ2Jhc2U2NCcpO1xuICAgICAgICAgIGlmIChpbWFnZUJhc2U2NCkge1xuICAgICAgICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsID0gYGRhdGE6JHt0aGlzLklNQUdFX01JTUV9O2Jhc2U2NCwke2ltYWdlQmFzZTY0fWA7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgc3RhdGljIGdldENvc1NpbWlsYXJpdHkoYTogbnVtYmVyW10sIGI6IG51bWJlcltdKSB7XG4gICAgaWYgKGNvc1NpbWlsYXJpdHlBKSB7XG4gICAgICByZXR1cm4gY29zU2ltaWxhcml0eUEoYSwgYik7XG4gICAgfVxuICAgIHJldHVybiBjb3NTaW1pbGFyaXR5QihhLCBiKTtcbiAgfVxuXG4gIHByaXZhdGUgcHVzaFBvc2VGcm9tU2ltaWxhclBvc2VRdWV1ZShuZXh0UG9zZVRpbWVNaWxpc2Vjb25kcz86IG51bWJlcikge1xuICAgIGlmICh0aGlzLnNpbWlsYXJQb3NlUXVldWUubGVuZ3RoID09PSAwKSByZXR1cm47XG5cbiAgICBpZiAodGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCA9PT0gMSkge1xuICAgICAgLy8g6aGe5Ly844Od44O844K644Kt44Ol44O844Gr44Od44O844K644GM5LiA44Gk44GX44GL44Gq44GE5aC05ZCI44CB5b2T6Kmy44Od44O844K644KS44Od44O844K66YWN5YiX44G46L+95YqgXG4gICAgICBjb25zdCBwb3NlID0gdGhpcy5zaW1pbGFyUG9zZVF1ZXVlWzBdO1xuICAgICAgdGhpcy5wb3Nlcy5wdXNoKHBvc2UpO1xuICAgICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlID0gW107XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgLy8g5ZCE44Od44O844K644Gu5oyB57aa5pmC6ZaT44KS6Kit5a6aXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLnNpbWlsYXJQb3NlUXVldWUubGVuZ3RoIC0gMTsgaSsrKSB7XG4gICAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWVbaV0uZHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZVtpICsgMV0udGltZU1pbGlzZWNvbmRzIC1cbiAgICAgICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlW2ldLnRpbWVNaWxpc2Vjb25kcztcbiAgICB9XG4gICAgaWYgKG5leHRQb3NlVGltZU1pbGlzZWNvbmRzKSB7XG4gICAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWVbXG4gICAgICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZS5sZW5ndGggLSAxXG4gICAgICBdLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICBuZXh0UG9zZVRpbWVNaWxpc2Vjb25kcyAtXG4gICAgICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZVt0aGlzLnNpbWlsYXJQb3NlUXVldWUubGVuZ3RoIC0gMV0udGltZU1pbGlzZWNvbmRzO1xuICAgIH1cblxuICAgIC8vIOmhnuS8vOODneODvOOCuuOCreODpeODvOOBruS4reOBi+OCieacgOOCguaMgee2muaZgumWk+OBjOmVt+OBhOODneODvOOCuuOCkumBuOaKnlxuICAgIGNvbnN0IHNlbGVjdGVkUG9zZSA9IFBvc2VTZXQuZ2V0U3VpdGFibGVQb3NlQnlQb3Nlcyh0aGlzLnNpbWlsYXJQb3NlUXVldWUpO1xuXG4gICAgLy8g6YG45oqe44GV44KM44Gq44GL44Gj44Gf44Od44O844K644KS5YiX5oyZXG4gICAgc2VsZWN0ZWRQb3NlLmRlYnVnLmR1cGxpY2F0ZWRJdGVtcyA9IHRoaXMuc2ltaWxhclBvc2VRdWV1ZVxuICAgICAgLmZpbHRlcigoaXRlbTogUG9zZVNldEl0ZW0pID0+IHtcbiAgICAgICAgcmV0dXJuIGl0ZW0udGltZU1pbGlzZWNvbmRzICE9PSBzZWxlY3RlZFBvc2UudGltZU1pbGlzZWNvbmRzO1xuICAgICAgfSlcbiAgICAgIC5tYXAoKGl0ZW06IFBvc2VTZXRJdGVtKSA9PiB7XG4gICAgICAgIHJldHVybiB7XG4gICAgICAgICAgdGltZU1pbGlzZWNvbmRzOiBpdGVtLnRpbWVNaWxpc2Vjb25kcyxcbiAgICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBpdGVtLmR1cmF0aW9uTWlsaXNlY29uZHMsXG4gICAgICAgICAgYm9keVNpbWlsYXJpdHk6IHVuZGVmaW5lZCxcbiAgICAgICAgICBoYW5kU2ltaWxhcml0eTogdW5kZWZpbmVkLFxuICAgICAgICB9O1xuICAgICAgfSk7XG4gICAgc2VsZWN0ZWRQb3NlLm1lcmdlZFRpbWVNaWxpc2Vjb25kcyA9XG4gICAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWVbMF0udGltZU1pbGlzZWNvbmRzO1xuICAgIHNlbGVjdGVkUG9zZS5tZXJnZWREdXJhdGlvbk1pbGlzZWNvbmRzID0gdGhpcy5zaW1pbGFyUG9zZVF1ZXVlLnJlZHVjZShcbiAgICAgIChzdW06IG51bWJlciwgaXRlbTogUG9zZVNldEl0ZW0pID0+IHtcbiAgICAgICAgcmV0dXJuIHN1bSArIGl0ZW0uZHVyYXRpb25NaWxpc2Vjb25kcztcbiAgICAgIH0sXG4gICAgICAwXG4gICAgKTtcblxuICAgIC8vIOW9k+ipsuODneODvOOCuuOCkuODneODvOOCuumFjeWIl+OBuOi/veWKoFxuICAgIGlmICh0aGlzLklTX0VOQUJMRURfUkVNT1ZFX0RVUExJQ0FURURfUE9TRVNfRk9SX0FST1VORCkge1xuICAgICAgdGhpcy5wb3Nlcy5wdXNoKHNlbGVjdGVkUG9zZSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIOODh+ODkOODg+OCsOeUqFxuICAgICAgdGhpcy5wb3Nlcy5wdXNoKC4uLnRoaXMuc2ltaWxhclBvc2VRdWV1ZSk7XG4gICAgfVxuXG4gICAgLy8g6aGe5Ly844Od44O844K644Kt44Ol44O844KS44Kv44Oq44KiXG4gICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlID0gW107XG4gIH1cblxuICByZW1vdmVEdXBsaWNhdGVkUG9zZXMoKTogdm9pZCB7XG4gICAgLy8g5YWo44Od44O844K644KS5q+U6LyD44GX44Gm6aGe5Ly844Od44O844K644KS5YmK6ZmkXG4gICAgY29uc3QgbmV3UG9zZXM6IFBvc2VTZXRJdGVtW10gPSBbXSxcbiAgICAgIHJlbW92ZWRQb3NlczogUG9zZVNldEl0ZW1bXSA9IFtdO1xuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgZHVwbGljYXRlZFBvc2U6IFBvc2VTZXRJdGVtO1xuICAgICAgZm9yIChjb25zdCBpbnNlcnRlZFBvc2Ugb2YgbmV3UG9zZXMpIHtcbiAgICAgICAgY29uc3QgaXNTaW1pbGFyQm9keVBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckJvZHlQb3NlKFxuICAgICAgICAgIHBvc2UuYm9keVZlY3RvcixcbiAgICAgICAgICBpbnNlcnRlZFBvc2UuYm9keVZlY3RvclxuICAgICAgICApO1xuICAgICAgICBjb25zdCBpc1NpbWlsYXJIYW5kUG9zZSA9XG4gICAgICAgICAgcG9zZS5oYW5kVmVjdG9yICYmIGluc2VydGVkUG9zZS5oYW5kVmVjdG9yXG4gICAgICAgICAgICA/IFBvc2VTZXQuaXNTaW1pbGFySGFuZFBvc2UoXG4gICAgICAgICAgICAgICAgcG9zZS5oYW5kVmVjdG9yLFxuICAgICAgICAgICAgICAgIGluc2VydGVkUG9zZS5oYW5kVmVjdG9yLFxuICAgICAgICAgICAgICAgIDAuOVxuICAgICAgICAgICAgICApXG4gICAgICAgICAgICA6IGZhbHNlO1xuXG4gICAgICAgIGlmIChpc1NpbWlsYXJCb2R5UG9zZSAmJiBpc1NpbWlsYXJIYW5kUG9zZSkge1xuICAgICAgICAgIC8vIOi6q+S9k+ODu+aJi+OBqOOCguOBq+mhnuS8vOODneODvOOCuuOBquOCieOBsFxuICAgICAgICAgIGR1cGxpY2F0ZWRQb3NlID0gaW5zZXJ0ZWRQb3NlO1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChkdXBsaWNhdGVkUG9zZSkge1xuICAgICAgICByZW1vdmVkUG9zZXMucHVzaChwb3NlKTtcbiAgICAgICAgaWYgKGR1cGxpY2F0ZWRQb3NlLmRlYnVnLmR1cGxpY2F0ZWRJdGVtcykge1xuICAgICAgICAgIGR1cGxpY2F0ZWRQb3NlLmRlYnVnLmR1cGxpY2F0ZWRJdGVtcy5wdXNoKHtcbiAgICAgICAgICAgIHRpbWVNaWxpc2Vjb25kczogcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBwb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMsXG4gICAgICAgICAgICBib2R5U2ltaWxhcml0eTogdW5kZWZpbmVkLFxuICAgICAgICAgICAgaGFuZFNpbWlsYXJpdHk6IHVuZGVmaW5lZCxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfVxuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cblxuICAgICAgbmV3UG9zZXMucHVzaChwb3NlKTtcbiAgICB9XG5cbiAgICBjb25zb2xlLmluZm8oXG4gICAgICBgW1Bvc2VTZXRdIHJlbW92ZUR1cGxpY2F0ZWRQb3NlcyAtIFJlZHVjZWQgJHt0aGlzLnBvc2VzLmxlbmd0aH0gcG9zZXMgLT4gJHtuZXdQb3Nlcy5sZW5ndGh9IHBvc2VzYCxcbiAgICAgIHtcbiAgICAgICAgcmVtb3ZlZDogcmVtb3ZlZFBvc2VzLFxuICAgICAgICBrZWVwZWQ6IG5ld1Bvc2VzLFxuICAgICAgfVxuICAgICk7XG4gICAgdGhpcy5wb3NlcyA9IG5ld1Bvc2VzO1xuICB9XG5cbiAgcHJpdmF0ZSBnZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKElNQUdFX01JTUU6IHN0cmluZykge1xuICAgIHN3aXRjaCAoSU1BR0VfTUlNRSkge1xuICAgICAgY2FzZSAnaW1hZ2UvcG5nJzpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgICAgY2FzZSAnaW1hZ2UvanBlZyc6XG4gICAgICAgIHJldHVybiAnanBnJztcbiAgICAgIGNhc2UgJ2ltYWdlL3dlYnAnOlxuICAgICAgICByZXR1cm4gJ3dlYnAnO1xuICAgICAgZGVmYXVsdDpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgIH1cbiAgfVxufVxuIl19