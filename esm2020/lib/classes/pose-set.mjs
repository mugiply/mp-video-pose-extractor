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
     * 指定されたID (PoseSetItemId) によるポーズの取得
     * @param poseSetItemId
     * @returns ポーズ
     */
    getPoseById(poseSetItemId) {
        if (this.poses === undefined)
            return undefined;
        return this.poses.find((pose) => pose.id === poseSetItemId);
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
            id: PoseSet.getIdByTimeMiliseconds(videoTimeMiliseconds),
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
            mergedTimeMiliseconds: videoTimeMiliseconds,
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
            if (this.poses[i].durationMiliseconds === -1) {
                this.poses[i].durationMiliseconds =
                    this.poses[i + 1].timeMiliseconds - this.poses[i].timeMiliseconds;
            }
            if (this.poses[i].mergedDurationMiliseconds === -1) {
                this.poses[i].mergedDurationMiliseconds =
                    this.poses[i].durationMiliseconds;
            }
        }
        if (this.poses[this.poses.length - 1].durationMiliseconds === -1) {
            this.poses[this.poses.length - 1].durationMiliseconds =
                this.videoMetadata.duration -
                    this.poses[this.poses.length - 1].timeMiliseconds;
        }
        if (this.poses[this.poses.length - 1].mergedDurationMiliseconds === -1) {
            this.poses[this.poses.length - 1].mergedDurationMiliseconds =
                this.poses[this.poses.length - 1].durationMiliseconds;
        }
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
                    jsZip.file(`frame-${pose.id}.${imageFileExt}`, base64, {
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
                    jsZip.file(`pose-${pose.id}.${imageFileExt}`, base64, {
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
                    jsZip.file(`face-${pose.id}.${imageFileExt}`, base64, {
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
                    id: pose.id,
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
                id: item.id === undefined
                    ? PoseSet.getIdByTimeMiliseconds(item.t)
                    : item.id,
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
                    let imageBase64;
                    if (zip.file(`frame-${pose.id}.${fileExt}`)) {
                        imageBase64 = await zip
                            .file(`frame-${pose.id}.${fileExt}`)
                            ?.async('base64');
                    }
                    else {
                        imageBase64 = await zip
                            .file(`frame-${pose.timeMiliseconds}.${fileExt}`)
                            ?.async('base64');
                    }
                    if (imageBase64) {
                        pose.frameImageDataUrl = `data:${this.IMAGE_MIME};base64,${imageBase64}`;
                    }
                }
                if (!pose.poseImageDataUrl) {
                    const poseImageFileName = `pose-${pose.id}.${fileExt}`;
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
                id: item.id,
                timeMiliseconds: item.timeMiliseconds,
                durationMiliseconds: item.durationMiliseconds,
            };
        });
        // 選択されたポーズの情報を更新
        selectedPose.mergedTimeMiliseconds =
            this.similarPoseQueue[0].timeMiliseconds;
        selectedPose.mergedDurationMiliseconds = this.similarPoseQueue.reduce((sum, item) => {
            return sum + item.durationMiliseconds;
        }, 0);
        selectedPose.id = PoseSet.getIdByTimeMiliseconds(selectedPose.mergedTimeMiliseconds);
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
                        id: pose.id,
                        timeMiliseconds: pose.timeMiliseconds,
                        durationMiliseconds: pose.durationMiliseconds,
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
    static getIdByTimeMiliseconds(timeMiliseconds) {
        return Math.floor(timeMiliseconds / 100) * 100;
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1zZXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi9wcm9qZWN0cy9uZ3gtbXAtcG9zZS1leHRyYWN0b3Ivc3JjL2xpYi9jbGFzc2VzL3Bvc2Utc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxjQUFjLEVBQVcsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssS0FBSyxNQUFNLE9BQU8sQ0FBQztBQU0vQixhQUFhO0FBQ2IsT0FBTyxjQUFjLE1BQU0sZ0JBQWdCLENBQUM7QUFDNUMsYUFBYTtBQUNiLE9BQU8sS0FBSyxjQUFjLE1BQU0sZ0JBQWdCLENBQUM7QUFHakQsT0FBTyxFQUFFLFlBQVksRUFBRSxNQUFNLDJCQUEyQixDQUFDO0FBR3pELE1BQU0sT0FBTyxPQUFPO0lBOEVsQjtRQXBFTyxVQUFLLEdBQWtCLEVBQUUsQ0FBQztRQUMxQixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQThDckMsaUJBQWlCO1FBQ1QscUJBQWdCLEdBQWtCLEVBQUUsQ0FBQztRQUU3Qyx1QkFBdUI7UUFDTixrREFBNkMsR0FBRyxJQUFJLENBQUM7UUFFdEUsYUFBYTtRQUNJLGdCQUFXLEdBQVcsSUFBSSxDQUFDO1FBQzNCLGVBQVUsR0FDekIsWUFBWSxDQUFDO1FBQ0Usa0JBQWEsR0FBRyxHQUFHLENBQUM7UUFFckMsVUFBVTtRQUNPLGdDQUEyQixHQUFHLFNBQVMsQ0FBQztRQUN4Qyx5Q0FBb0MsR0FBRyxFQUFFLENBQUM7UUFFM0QsV0FBVztRQUNNLHVDQUFrQyxHQUFHLFNBQVMsQ0FBQztRQUMvQyx1Q0FBa0MsR0FBRyxXQUFXLENBQUM7UUFDakQsNENBQXVDLEdBQUcsR0FBRyxDQUFDO1FBRzdELElBQUksQ0FBQyxhQUFhLEdBQUc7WUFDbkIsSUFBSSxFQUFFLEVBQUU7WUFDUixLQUFLLEVBQUUsQ0FBQztZQUNSLE1BQU0sRUFBRSxDQUFDO1lBQ1QsUUFBUSxFQUFFLENBQUM7WUFDWCxx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Iiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgUE9TRV9MQU5ETUFSS1MsIFJlc3VsdHMgfSBmcm9tICdAbWVkaWFwaXBlL2hvbGlzdGljJztcbmltcG9ydCAqIGFzIEpTWmlwIGZyb20gJ2pzemlwJztcbmltcG9ydCB7IFBvc2VTZXRJdGVtIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXNldC1pdGVtJztcbmltcG9ydCB7IFBvc2VTZXRKc29uIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXNldC1qc29uJztcbmltcG9ydCB7IFBvc2VTZXRKc29uSXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1zZXQtanNvbi1pdGVtJztcbmltcG9ydCB7IEJvZHlWZWN0b3IgfSBmcm9tICcuLi9pbnRlcmZhY2VzL2JvZHktdmVjdG9yJztcblxuLy8gQHRzLWlnbm9yZVxuaW1wb3J0IGNvc1NpbWlsYXJpdHlBIGZyb20gJ2Nvcy1zaW1pbGFyaXR5Jztcbi8vIEB0cy1pZ25vcmVcbmltcG9ydCAqIGFzIGNvc1NpbWlsYXJpdHlCIGZyb20gJ2Nvcy1zaW1pbGFyaXR5JztcblxuaW1wb3J0IHsgU2ltaWxhclBvc2VJdGVtIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9zaW1pbGFyLXBvc2UtaXRlbSc7XG5pbXBvcnQgeyBJbWFnZVRyaW1tZXIgfSBmcm9tICcuL2ludGVybmFscy9pbWFnZS10cmltbWVyJztcbmltcG9ydCB7IEhhbmRWZWN0b3IgfSBmcm9tICcuLi9pbnRlcmZhY2VzL2hhbmQtdmVjdG9yJztcblxuZXhwb3J0IGNsYXNzIFBvc2VTZXQge1xuICBwdWJsaWMgZ2VuZXJhdG9yPzogc3RyaW5nO1xuICBwdWJsaWMgdmVyc2lvbj86IG51bWJlcjtcbiAgcHJpdmF0ZSB2aWRlb01ldGFkYXRhIToge1xuICAgIG5hbWU6IHN0cmluZztcbiAgICB3aWR0aDogbnVtYmVyO1xuICAgIGhlaWdodDogbnVtYmVyO1xuICAgIGR1cmF0aW9uOiBudW1iZXI7XG4gICAgZmlyc3RQb3NlRGV0ZWN0ZWRUaW1lOiBudW1iZXI7XG4gIH07XG4gIHB1YmxpYyBwb3NlczogUG9zZVNldEl0ZW1bXSA9IFtdO1xuICBwdWJsaWMgaXNGaW5hbGl6ZWQ/OiBib29sZWFuID0gZmFsc2U7XG5cbiAgLy8gQm9keVZlY3RvciDjga7jgq3jg7zlkI1cbiAgcHVibGljIHN0YXRpYyByZWFkb25seSBCT0RZX1ZFQ1RPUl9NQVBQSU5HUyA9IFtcbiAgICAvLyDlj7PohZVcbiAgICAncmlnaHRXcmlzdFRvUmlnaHRFbGJvdycsXG4gICAgJ3JpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXInLFxuICAgIC8vIOW3puiFlVxuICAgICdsZWZ0V3Jpc3RUb0xlZnRFbGJvdycsXG4gICAgJ2xlZnRFbGJvd1RvTGVmdFNob3VsZGVyJyxcbiAgXTtcblxuICAvLyBIYW5kVmVjdG9yIOOBruOCreODvOWQjVxuICBwdWJsaWMgc3RhdGljIHJlYWRvbmx5IEhBTkRfVkVDVE9SX01BUFBJTkdTID0gW1xuICAgIC8vIOWPs+aJiyAtIOimquaMh1xuICAgICdyaWdodFRodW1iVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDkurrlt67jgZfmjIdcbiAgICAncmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ3JpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5Y+z5omLIC0g5Lit5oyHXG4gICAgJ3JpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5Y+z5omLIC0g6Jas5oyHXG4gICAgJ3JpZ2h0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ3JpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDlsI/mjIdcbiAgICAncmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ3JpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g6Kaq5oyHXG4gICAgJ2xlZnRUaHVtYlRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlt6bmiYsgLSDkurrlt67jgZfmjIdcbiAgICAnbGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAnbGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOS4reaMh1xuICAgICdsZWZ0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAnbGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlt6bmiYsgLSDolqzmjIdcbiAgICAnbGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlt6bmiYsgLSDlsI/mjIdcbiAgICAnbGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAnbGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICBdO1xuXG4gIC8vIOODneODvOOCuuOCkui/veWKoOOBmeOCi+OBn+OCgeOBruOCreODpeODvFxuICBwcml2YXRlIHNpbWlsYXJQb3NlUXVldWU6IFBvc2VTZXRJdGVtW10gPSBbXTtcblxuICAvLyDpoZ7kvLzjg53jg7zjgrrjga7pmaTljrsgLSDlkITjg53jg7zjgrrjga7liY3lvozjgYvjgolcbiAgcHJpdmF0ZSByZWFkb25seSBJU19FTkFCTEVEX1JFTU9WRV9EVVBMSUNBVEVEX1BPU0VTX0ZPUl9BUk9VTkQgPSB0cnVlO1xuXG4gIC8vIOeUu+WDj+abuOOBjeWHuuOBl+aZguOBruioreWumlxuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX1dJRFRIOiBudW1iZXIgPSAxMDgwO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX01JTUU6ICdpbWFnZS9qcGVnJyB8ICdpbWFnZS9wbmcnIHwgJ2ltYWdlL3dlYnAnID1cbiAgICAnaW1hZ2Uvd2VicCc7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfUVVBTElUWSA9IDAuODtcblxuICAvLyDnlLvlg4/jga7kvZnnmb3pmaTljrtcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9NQVJHSU5fVFJJTU1JTkdfQ09MT1IgPSAnIzAwMDAwMCc7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0RJRkZfVEhSRVNIT0xEID0gNTA7XG5cbiAgLy8g55S75YOP44Gu6IOM5pmv6Imy572u5o+bXG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX1NSQ19DT0xPUiA9ICcjMDE2QUZEJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRFNUX0NPTE9SID0gJyNGRkZGRkYwMCc7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RJRkZfVEhSRVNIT0xEID0gMTMwO1xuXG4gIGNvbnN0cnVjdG9yKCkge1xuICAgIHRoaXMudmlkZW9NZXRhZGF0YSA9IHtcbiAgICAgIG5hbWU6ICcnLFxuICAgICAgd2lkdGg6IDAsXG4gICAgICBoZWlnaHQ6IDAsXG4gICAgICBkdXJhdGlvbjogMCxcbiAgICAgIGZpcnN0UG9zZURldGVjdGVkVGltZTogMCxcbiAgICB9O1xuICB9XG5cbiAgZ2V0VmlkZW9OYW1lKCkge1xuICAgIHJldHVybiB0aGlzLnZpZGVvTWV0YWRhdGEubmFtZTtcbiAgfVxuXG4gIHNldFZpZGVvTmFtZSh2aWRlb05hbWU6IHN0cmluZykge1xuICAgIHRoaXMudmlkZW9NZXRhZGF0YS5uYW1lID0gdmlkZW9OYW1lO1xuICB9XG5cbiAgc2V0VmlkZW9NZXRhRGF0YSh3aWR0aDogbnVtYmVyLCBoZWlnaHQ6IG51bWJlciwgZHVyYXRpb246IG51bWJlcikge1xuICAgIHRoaXMudmlkZW9NZXRhZGF0YS53aWR0aCA9IHdpZHRoO1xuICAgIHRoaXMudmlkZW9NZXRhZGF0YS5oZWlnaHQgPSBoZWlnaHQ7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLmR1cmF0aW9uID0gZHVyYXRpb247XG4gIH1cblxuICAvKipcbiAgICog44Od44O844K65pWw44Gu5Y+W5b6XXG4gICAqIEByZXR1cm5zXG4gICAqL1xuICBnZXROdW1iZXJPZlBvc2VzKCk6IG51bWJlciB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIC0xO1xuICAgIHJldHVybiB0aGlzLnBvc2VzLmxlbmd0aDtcbiAgfVxuXG4gIC8qKlxuICAgKiDlhajjg53jg7zjgrrjga7lj5blvpdcbiAgICogQHJldHVybnMg5YWo44Gm44Gu44Od44O844K6XG4gICAqL1xuICBnZXRQb3NlcygpOiBQb3NlU2V0SXRlbVtdIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gW107XG4gICAgcmV0dXJuIHRoaXMucG9zZXM7XG4gIH1cblxuICAvKipcbiAgICog5oyH5a6a44GV44KM44GfSUQgKFBvc2VTZXRJdGVtSWQpIOOBq+OCiOOCi+ODneODvOOCuuOBruWPluW+l1xuICAgKiBAcGFyYW0gcG9zZVNldEl0ZW1JZFxuICAgKiBAcmV0dXJucyDjg53jg7zjgrpcbiAgICovXG4gIGdldFBvc2VCeUlkKHBvc2VTZXRJdGVtSWQ6IG51bWJlcik6IFBvc2VTZXRJdGVtIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgIHJldHVybiB0aGlzLnBvc2VzLmZpbmQoKHBvc2UpID0+IHBvc2UuaWQgPT09IHBvc2VTZXRJdGVtSWQpO1xuICB9XG5cbiAgLyoqXG4gICAqIOaMh+WumuOBleOCjOOBn+aZgumWk+OBq+OCiOOCi+ODneODvOOCuuOBruWPluW+l1xuICAgKiBAcGFyYW0gdGltZU1pbGlzZWNvbmRzIOODneODvOOCuuOBruaZgumWkyAo44Of44Oq56eSKVxuICAgKiBAcmV0dXJucyDjg53jg7zjgrpcbiAgICovXG4gIGdldFBvc2VCeVRpbWUodGltZU1pbGlzZWNvbmRzOiBudW1iZXIpOiBQb3NlU2V0SXRlbSB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICByZXR1cm4gdGhpcy5wb3Nlcy5maW5kKChwb3NlKSA9PiBwb3NlLnRpbWVNaWxpc2Vjb25kcyA9PT0gdGltZU1pbGlzZWNvbmRzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiDjg53jg7zjgrrjga7ov73liqBcbiAgICovXG4gIHB1c2hQb3NlKFxuICAgIHZpZGVvVGltZU1pbGlzZWNvbmRzOiBudW1iZXIsXG4gICAgZnJhbWVJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICBwb3NlSW1hZ2VEYXRhVXJsOiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gICAgZmFjZUZyYW1lSW1hZ2VEYXRhVXJsOiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gICAgcmVzdWx0czogUmVzdWx0c1xuICApOiBQb3NlU2V0SXRlbSB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKHRoaXMucG9zZXMubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZmlyc3RQb3NlRGV0ZWN0ZWRUaW1lID0gdmlkZW9UaW1lTWlsaXNlY29uZHM7XG4gICAgfVxuXG4gICAgaWYgKHJlc3VsdHMucG9zZUxhbmRtYXJrcyA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHBvc2VMYW5kbWFya3NgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlOiBhbnlbXSA9IChyZXN1bHRzIGFzIGFueSkuZWFcbiAgICAgID8gKHJlc3VsdHMgYXMgYW55KS5lYVxuICAgICAgOiBbXTtcbiAgICBpZiAocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubGVuZ3RoID09PSAwKSB7XG4gICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBwb3NlIHdpdGggdGhlIHdvcmxkIGNvb3JkaW5hdGVgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IGJvZHlWZWN0b3IgPSBQb3NlU2V0LmdldEJvZHlWZWN0b3IocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUpO1xuICAgIGlmICghYm9keVZlY3Rvcikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBib2R5IHZlY3RvcmAsXG4gICAgICAgIHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGlmIChcbiAgICAgIHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCAmJlxuICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZFxuICAgICkge1xuICAgICAgY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgaGFuZCBsYW5kbWFya3NgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH0gZWxzZSBpZiAocmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBsZWZ0IGhhbmQgbGFuZG1hcmtzYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHtcbiAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIHJpZ2h0IGhhbmQgbGFuZG1hcmtzYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICB9XG5cbiAgICBjb25zdCBoYW5kVmVjdG9yID0gUG9zZVNldC5nZXRIYW5kVmVjdG9yKFxuICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyxcbiAgICAgIHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzXG4gICAgKTtcbiAgICBpZiAoIWhhbmRWZWN0b3IpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgaGFuZCB2ZWN0b3JgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2U6IFBvc2VTZXRJdGVtID0ge1xuICAgICAgaWQ6IFBvc2VTZXQuZ2V0SWRCeVRpbWVNaWxpc2Vjb25kcyh2aWRlb1RpbWVNaWxpc2Vjb25kcyksXG4gICAgICB0aW1lTWlsaXNlY29uZHM6IHZpZGVvVGltZU1pbGlzZWNvbmRzLFxuICAgICAgZHVyYXRpb25NaWxpc2Vjb25kczogLTEsXG4gICAgICBwb3NlOiBwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZS5tYXAoKHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsueCxcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay55LFxuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLnosXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsudmlzaWJpbGl0eSxcbiAgICAgICAgXTtcbiAgICAgIH0pLFxuICAgICAgbGVmdEhhbmQ6IHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3M/Lm1hcCgobm9ybWFsaXplZExhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLngsXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnksXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnosXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIHJpZ2h0SGFuZDogcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3M/Lm1hcCgobm9ybWFsaXplZExhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLngsXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnksXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnosXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIGJvZHlWZWN0b3I6IGJvZHlWZWN0b3IsXG4gICAgICBoYW5kVmVjdG9yOiBoYW5kVmVjdG9yLFxuICAgICAgZnJhbWVJbWFnZURhdGFVcmw6IGZyYW1lSW1hZ2VEYXRhVXJsLFxuICAgICAgcG9zZUltYWdlRGF0YVVybDogcG9zZUltYWdlRGF0YVVybCxcbiAgICAgIGZhY2VGcmFtZUltYWdlRGF0YVVybDogZmFjZUZyYW1lSW1hZ2VEYXRhVXJsLFxuICAgICAgZXh0ZW5kZWREYXRhOiB7fSxcbiAgICAgIGRlYnVnOiB7XG4gICAgICAgIGR1cGxpY2F0ZWRJdGVtczogW10sXG4gICAgICB9LFxuICAgICAgbWVyZ2VkVGltZU1pbGlzZWNvbmRzOiB2aWRlb1RpbWVNaWxpc2Vjb25kcyxcbiAgICAgIG1lcmdlZER1cmF0aW9uTWlsaXNlY29uZHM6IC0xLFxuICAgIH07XG5cbiAgICBsZXQgbGFzdFBvc2U7XG4gICAgaWYgKHRoaXMucG9zZXMubGVuZ3RoID09PSAwICYmIDEgPD0gdGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCkge1xuICAgICAgLy8g6aGe5Ly844Od44O844K644Kt44Ol44O844GL44KJ5pyA5b6M44Gu44Od44O844K644KS5Y+W5b6XXG4gICAgICBsYXN0UG9zZSA9IHRoaXMuc2ltaWxhclBvc2VRdWV1ZVt0aGlzLnNpbWlsYXJQb3NlUXVldWUubGVuZ3RoIC0gMV07XG4gICAgfSBlbHNlIGlmICgxIDw9IHRoaXMucG9zZXMubGVuZ3RoKSB7XG4gICAgICAvLyDjg53jg7zjgrrphY3liJfjgYvjgonmnIDlvozjga7jg53jg7zjgrrjgpLlj5blvpdcbiAgICAgIGxhc3RQb3NlID0gdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdO1xuICAgIH1cblxuICAgIGlmIChsYXN0UG9zZSkge1xuICAgICAgLy8g5pyA5b6M44Gu44Od44O844K644GM44GC44KM44Gw44CB6aGe5Ly844Od44O844K644GL44Gp44GG44GL44KS5q+U6LyDXG4gICAgICBjb25zdCBpc1NpbWlsYXJCb2R5UG9zZSA9IFBvc2VTZXQuaXNTaW1pbGFyQm9keVBvc2UoXG4gICAgICAgIHBvc2UuYm9keVZlY3RvcixcbiAgICAgICAgbGFzdFBvc2UuYm9keVZlY3RvclxuICAgICAgKTtcblxuICAgICAgbGV0IGlzU2ltaWxhckhhbmRQb3NlID0gdHJ1ZTtcbiAgICAgIGlmIChsYXN0UG9zZS5oYW5kVmVjdG9yICYmIHBvc2UuaGFuZFZlY3Rvcikge1xuICAgICAgICBpc1NpbWlsYXJIYW5kUG9zZSA9IFBvc2VTZXQuaXNTaW1pbGFySGFuZFBvc2UoXG4gICAgICAgICAgcG9zZS5oYW5kVmVjdG9yLFxuICAgICAgICAgIGxhc3RQb3NlLmhhbmRWZWN0b3JcbiAgICAgICAgKTtcbiAgICAgIH0gZWxzZSBpZiAoIWxhc3RQb3NlLmhhbmRWZWN0b3IgJiYgcG9zZS5oYW5kVmVjdG9yKSB7XG4gICAgICAgIGlzU2ltaWxhckhhbmRQb3NlID0gZmFsc2U7XG4gICAgICB9XG5cbiAgICAgIGlmICghaXNTaW1pbGFyQm9keVBvc2UgfHwgIWlzU2ltaWxhckhhbmRQb3NlKSB7XG4gICAgICAgIC8vIOi6q+S9k+ODu+aJi+OBruOBhOOBmuOCjOOBi+OBjOWJjeOBruODneODvOOCuuOBqOmhnuS8vOOBl+OBpuOBhOOBquOBhOOBquOCieOBsOOAgemhnuS8vOODneODvOOCuuOCreODpeODvOOCkuWHpueQhuOBl+OBpuOAgeODneODvOOCuumFjeWIl+OBuOi/veWKoFxuICAgICAgICB0aGlzLnB1c2hQb3NlRnJvbVNpbWlsYXJQb3NlUXVldWUocG9zZS50aW1lTWlsaXNlY29uZHMpO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIOmhnuS8vOODneODvOOCuuOCreODpeODvOOBuOi/veWKoFxuICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZS5wdXNoKHBvc2UpO1xuXG4gICAgcmV0dXJuIHBvc2U7XG4gIH1cblxuICAvKipcbiAgICog44Od44O844K644Gu6YWN5YiX44GL44KJ44Od44O844K644GM5rG644G+44Gj44Gm44GE44KL556s6ZaT44KS5Y+W5b6XXG4gICAqIEBwYXJhbSBwb3NlcyDjg53jg7zjgrrjga7phY3liJdcbiAgICogQHJldHVybnMg44Od44O844K644GM5rG644G+44Gj44Gm44GE44KL556s6ZaTXG4gICAqL1xuICBzdGF0aWMgZ2V0U3VpdGFibGVQb3NlQnlQb3Nlcyhwb3NlczogUG9zZVNldEl0ZW1bXSk6IFBvc2VTZXRJdGVtIHtcbiAgICBpZiAocG9zZXMubGVuZ3RoID09PSAwKSByZXR1cm4gbnVsbDtcbiAgICBpZiAocG9zZXMubGVuZ3RoID09PSAxKSB7XG4gICAgICByZXR1cm4gcG9zZXNbMV07XG4gICAgfVxuXG4gICAgLy8g5ZCE5qiZ5pys44Od44O844K644GU44Go44Gu6aGe5Ly85bqm44KS5Yid5pyf5YyWXG4gICAgY29uc3Qgc2ltaWxhcml0aWVzT2ZQb3Nlczoge1xuICAgICAgW2tleTogbnVtYmVyXToge1xuICAgICAgICBoYW5kU2ltaWxhcml0eTogbnVtYmVyO1xuICAgICAgICBib2R5U2ltaWxhcml0eTogbnVtYmVyO1xuICAgICAgfVtdO1xuICAgIH0gPSB7fTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHBvc2VzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBzaW1pbGFyaXRpZXNPZlBvc2VzW3Bvc2VzW2ldLnRpbWVNaWxpc2Vjb25kc10gPSBwb3Nlcy5tYXAoXG4gICAgICAgIChwb3NlOiBQb3NlU2V0SXRlbSkgPT4ge1xuICAgICAgICAgIHJldHVybiB7XG4gICAgICAgICAgICBoYW5kU2ltaWxhcml0eTogMCxcbiAgICAgICAgICAgIGJvZHlTaW1pbGFyaXR5OiAwLFxuICAgICAgICAgIH07XG4gICAgICAgIH1cbiAgICAgICk7XG4gICAgfVxuXG4gICAgLy8g5ZCE5qiZ5pys44Od44O844K644GU44Go44Gu6aGe5Ly85bqm44KS6KiI566XXG4gICAgZm9yIChsZXQgc2FtcGxlUG9zZSBvZiBwb3Nlcykge1xuICAgICAgbGV0IGhhbmRTaW1pbGFyaXR5OiBudW1iZXI7XG5cbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcG9zZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgY29uc3QgcG9zZSA9IHBvc2VzW2ldO1xuICAgICAgICBpZiAocG9zZS5oYW5kVmVjdG9yICYmIHNhbXBsZVBvc2UuaGFuZFZlY3Rvcikge1xuICAgICAgICAgIGhhbmRTaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRIYW5kU2ltaWxhcml0eShcbiAgICAgICAgICAgIHBvc2UuaGFuZFZlY3RvcixcbiAgICAgICAgICAgIHNhbXBsZVBvc2UuaGFuZFZlY3RvclxuICAgICAgICAgICk7XG4gICAgICAgIH1cblxuICAgICAgICBsZXQgYm9keVNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEJvZHlQb3NlU2ltaWxhcml0eShcbiAgICAgICAgICBwb3NlLmJvZHlWZWN0b3IsXG4gICAgICAgICAgc2FtcGxlUG9zZS5ib2R5VmVjdG9yXG4gICAgICAgICk7XG5cbiAgICAgICAgc2ltaWxhcml0aWVzT2ZQb3Nlc1tzYW1wbGVQb3NlLnRpbWVNaWxpc2Vjb25kc11baV0gPSB7XG4gICAgICAgICAgaGFuZFNpbWlsYXJpdHk6IGhhbmRTaW1pbGFyaXR5ID8/IDAsXG4gICAgICAgICAgYm9keVNpbWlsYXJpdHksXG4gICAgICAgIH07XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8g6aGe5Ly85bqm44Gu6auY44GE44OV44Os44O844Og44GM5aSa44GL44Gj44Gf44Od44O844K644KS6YG45oqeXG4gICAgY29uc3Qgc2ltaWxhcml0aWVzT2ZTYW1wbGVQb3NlcyA9IHBvc2VzLm1hcCgocG9zZTogUG9zZVNldEl0ZW0pID0+IHtcbiAgICAgIHJldHVybiBzaW1pbGFyaXRpZXNPZlBvc2VzW3Bvc2UudGltZU1pbGlzZWNvbmRzXS5yZWR1Y2UoXG4gICAgICAgIChcbiAgICAgICAgICBwcmV2OiBudW1iZXIsXG4gICAgICAgICAgY3VycmVudDogeyBoYW5kU2ltaWxhcml0eTogbnVtYmVyOyBib2R5U2ltaWxhcml0eTogbnVtYmVyIH1cbiAgICAgICAgKSA9PiB7XG4gICAgICAgICAgcmV0dXJuIHByZXYgKyBjdXJyZW50LmhhbmRTaW1pbGFyaXR5ICsgY3VycmVudC5ib2R5U2ltaWxhcml0eTtcbiAgICAgICAgfSxcbiAgICAgICAgMFxuICAgICAgKTtcbiAgICB9KTtcbiAgICBjb25zdCBtYXhTaW1pbGFyaXR5ID0gTWF0aC5tYXgoLi4uc2ltaWxhcml0aWVzT2ZTYW1wbGVQb3Nlcyk7XG4gICAgY29uc3QgbWF4U2ltaWxhcml0eUluZGV4ID0gc2ltaWxhcml0aWVzT2ZTYW1wbGVQb3Nlcy5pbmRleE9mKG1heFNpbWlsYXJpdHkpO1xuICAgIGNvbnN0IHNlbGVjdGVkUG9zZSA9IHBvc2VzW21heFNpbWlsYXJpdHlJbmRleF07XG4gICAgaWYgKCFzZWxlY3RlZFBvc2UpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBnZXRTdWl0YWJsZVBvc2VCeVBvc2VzYCxcbiAgICAgICAgc2ltaWxhcml0aWVzT2ZTYW1wbGVQb3NlcyxcbiAgICAgICAgbWF4U2ltaWxhcml0eSxcbiAgICAgICAgbWF4U2ltaWxhcml0eUluZGV4XG4gICAgICApO1xuICAgIH1cblxuICAgIGNvbnNvbGUuZGVidWcoYFtQb3NlU2V0XSBnZXRTdWl0YWJsZVBvc2VCeVBvc2VzYCwge1xuICAgICAgc2VsZWN0ZWQ6IHNlbGVjdGVkUG9zZSxcbiAgICAgIHVuc2VsZWN0ZWQ6IHBvc2VzLmZpbHRlcigocG9zZTogUG9zZVNldEl0ZW0pID0+IHtcbiAgICAgICAgcmV0dXJuIHBvc2UudGltZU1pbGlzZWNvbmRzICE9PSBzZWxlY3RlZFBvc2UudGltZU1pbGlzZWNvbmRzO1xuICAgICAgfSksXG4gICAgfSk7XG4gICAgcmV0dXJuIHNlbGVjdGVkUG9zZTtcbiAgfVxuXG4gIC8qKlxuICAgKiDmnIDntYLlh6bnkIZcbiAgICogKOmHjeikh+OBl+OBn+ODneODvOOCuuOBrumZpOWOu+OAgeeUu+WDj+OBruODnuODvOOCuOODs+mZpOWOu+OBquOBqSlcbiAgICovXG4gIGFzeW5jIGZpbmFsaXplKGlzUmVtb3ZlRHVwbGljYXRlOiBib29sZWFuID0gdHJ1ZSkge1xuICAgIGlmICh0aGlzLnNpbWlsYXJQb3NlUXVldWUubGVuZ3RoID4gMCkge1xuICAgICAgLy8g6aGe5Ly844Od44O844K644Kt44Ol44O844Gr44Od44O844K644GM5q6L44Gj44Gm44GE44KL5aC05ZCI44CB5pyA6YGp44Gq44Od44O844K644KS6YG45oqe44GX44Gm44Od44O844K66YWN5YiX44G46L+95YqgXG4gICAgICB0aGlzLnB1c2hQb3NlRnJvbVNpbWlsYXJQb3NlUXVldWUodGhpcy52aWRlb01ldGFkYXRhLmR1cmF0aW9uKTtcbiAgICB9XG5cbiAgICBpZiAoMCA9PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgLy8g44Od44O844K644GM5LiA44Gk44KC44Gq44GE5aC05ZCI44CB5Yem55CG44KS57WC5LqGXG4gICAgICB0aGlzLmlzRmluYWxpemVkID0gdHJ1ZTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICAvLyDjg53jg7zjgrrjga7mjIHntprmmYLplpPjgpLoqK3lrppcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMucG9zZXMubGVuZ3RoIC0gMTsgaSsrKSB7XG4gICAgICBpZiAodGhpcy5wb3Nlc1tpXS5kdXJhdGlvbk1pbGlzZWNvbmRzID09PSAtMSkge1xuICAgICAgICB0aGlzLnBvc2VzW2ldLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICAgIHRoaXMucG9zZXNbaSArIDFdLnRpbWVNaWxpc2Vjb25kcyAtIHRoaXMucG9zZXNbaV0udGltZU1pbGlzZWNvbmRzO1xuICAgICAgfVxuICAgICAgaWYgKHRoaXMucG9zZXNbaV0ubWVyZ2VkRHVyYXRpb25NaWxpc2Vjb25kcyA9PT0gLTEpIHtcbiAgICAgICAgdGhpcy5wb3Nlc1tpXS5tZXJnZWREdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgICB0aGlzLnBvc2VzW2ldLmR1cmF0aW9uTWlsaXNlY29uZHM7XG4gICAgICB9XG4gICAgfVxuXG4gICAgaWYgKHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXS5kdXJhdGlvbk1pbGlzZWNvbmRzID09PSAtMSkge1xuICAgICAgdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24gLVxuICAgICAgICB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV0udGltZU1pbGlzZWNvbmRzO1xuICAgIH1cbiAgICBpZiAodGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdLm1lcmdlZER1cmF0aW9uTWlsaXNlY29uZHMgPT09IC0xKSB7XG4gICAgICB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV0ubWVyZ2VkRHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgIHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXS5kdXJhdGlvbk1pbGlzZWNvbmRzO1xuICAgIH1cblxuICAgIC8vIOWFqOS9k+OBi+OCiemHjeikh+ODneODvOOCuuOCkumZpOWOu1xuICAgIGlmIChpc1JlbW92ZUR1cGxpY2F0ZSkge1xuICAgICAgdGhpcy5yZW1vdmVEdXBsaWNhdGVkUG9zZXMoKTtcbiAgICB9XG5cbiAgICAvLyDmnIDliJ3jga7jg53jg7zjgrrjgpLpmaTljrtcbiAgICB0aGlzLnBvc2VzLnNoaWZ0KCk7XG5cbiAgICAvLyDnlLvlg4/jga7jg57jg7zjgrjjg7PjgpLlj5blvpdcbiAgICBjb25zb2xlLmRlYnVnKGBbUG9zZVNldF0gZmluYWxpemUgLSBEZXRlY3RpbmcgaW1hZ2UgbWFyZ2lucy4uLmApO1xuICAgIGxldCBpbWFnZVRyaW1taW5nOlxuICAgICAgfCB7XG4gICAgICAgICAgbWFyZ2luVG9wOiBudW1iZXI7XG4gICAgICAgICAgbWFyZ2luQm90dG9tOiBudW1iZXI7XG4gICAgICAgICAgaGVpZ2h0TmV3OiBudW1iZXI7XG4gICAgICAgICAgaGVpZ2h0T2xkOiBudW1iZXI7XG4gICAgICAgICAgd2lkdGg6IG51bWJlcjtcbiAgICAgICAgfVxuICAgICAgfCB1bmRlZmluZWQgPSB1bmRlZmluZWQ7XG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGxldCBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgY29uc3QgbWFyZ2luQ29sb3IgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0TWFyZ2luQ29sb3IoKTtcbiAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBEZXRlY3RlZCBtYXJnaW4gY29sb3IuLi5gLFxuICAgICAgICBwb3NlLnRpbWVNaWxpc2Vjb25kcyxcbiAgICAgICAgbWFyZ2luQ29sb3JcbiAgICAgICk7XG4gICAgICBpZiAobWFyZ2luQ29sb3IgPT09IG51bGwpIGNvbnRpbnVlO1xuICAgICAgaWYgKG1hcmdpbkNvbG9yICE9PSB0aGlzLklNQUdFX01BUkdJTl9UUklNTUlOR19DT0xPUikge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHRyaW1tZWQgPSBhd2FpdCBpbWFnZVRyaW1tZXIudHJpbU1hcmdpbihcbiAgICAgICAgbWFyZ2luQ29sb3IsXG4gICAgICAgIHRoaXMuSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0RJRkZfVEhSRVNIT0xEXG4gICAgICApO1xuICAgICAgaWYgKCF0cmltbWVkKSBjb250aW51ZTtcbiAgICAgIGltYWdlVHJpbW1pbmcgPSB0cmltbWVkO1xuICAgICAgY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVybWluZWQgaW1hZ2UgdHJpbW1pbmcgcG9zaXRpb25zLi4uYCxcbiAgICAgICAgdHJpbW1lZFxuICAgICAgKTtcbiAgICAgIGJyZWFrO1xuICAgIH1cblxuICAgIC8vIOeUu+WDj+OCkuaVtOW9olxuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsIHx8ICFwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG5cbiAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBQcm9jZXNzaW5nIGltYWdlLi4uYCxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHNcbiAgICAgICk7XG5cbiAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOODleODrOODvOODoOeUu+WDj1xuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5mcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGlmIChpbWFnZVRyaW1taW5nKSB7XG4gICAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5jcm9wKFxuICAgICAgICAgIDAsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5tYXJnaW5Ub3AsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy53aWR0aCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLmhlaWdodE5ld1xuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVwbGFjZUNvbG9yKFxuICAgICAgICB0aGlzLklNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9TUkNfQ09MT1IsXG4gICAgICAgIHRoaXMuSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RTVF9DT0xPUixcbiAgICAgICAgdGhpcy5JTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRElGRl9USFJFU0hPTERcbiAgICAgICk7XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXNpemVXaXRoRml0KHtcbiAgICAgICAgd2lkdGg6IHRoaXMuSU1BR0VfV0lEVEgsXG4gICAgICB9KTtcblxuICAgICAgbGV0IG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgID8gdGhpcy5JTUFHRV9RVUFMSVRZXG4gICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICk7XG4gICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgZnJhbWUgaW1hZ2VgXG4gICAgICAgICk7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG5cbiAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOODneODvOOCuuODl+ODrOODk+ODpeODvOeUu+WDj1xuICAgICAgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5wb3NlSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgaWYgKGltYWdlVHJpbW1pbmcpIHtcbiAgICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmNyb3AoXG4gICAgICAgICAgMCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLm1hcmdpblRvcCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLndpZHRoLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcuaGVpZ2h0TmV3XG4gICAgICAgICk7XG4gICAgICB9XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXNpemVXaXRoRml0KHtcbiAgICAgICAgd2lkdGg6IHRoaXMuSU1BR0VfV0lEVEgsXG4gICAgICB9KTtcblxuICAgICAgbmV3RGF0YVVybCA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXREYXRhVXJsKFxuICAgICAgICB0aGlzLklNQUdFX01JTUUsXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL2pwZWcnIHx8IHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL3dlYnAnXG4gICAgICAgICAgPyB0aGlzLklNQUdFX1FVQUxJVFlcbiAgICAgICAgICA6IHVuZGVmaW5lZFxuICAgICAgKTtcbiAgICAgIGlmICghbmV3RGF0YVVybCkge1xuICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIENvdWxkIG5vdCBnZXQgdGhlIG5ldyBkYXRhdXJsIGZvciBwb3NlIHByZXZpZXcgaW1hZ2VgXG4gICAgICAgICk7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsID0gbmV3RGF0YVVybDtcblxuICAgICAgaWYgKHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOmhlOODleODrOODvOODoOeUu+WDj1xuICAgICAgICBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgICBuZXdEYXRhVXJsID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldERhdGFVcmwoXG4gICAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICAgIHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL2pwZWcnIHx8IHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL3dlYnAnXG4gICAgICAgICAgICA/IHRoaXMuSU1BR0VfUVVBTElUWVxuICAgICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICAgKTtcbiAgICAgICAgaWYgKCFuZXdEYXRhVXJsKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIENvdWxkIG5vdCBnZXQgdGhlIG5ldyBkYXRhdXJsIGZvciBmYWNlIGZyYW1lIGltYWdlYFxuICAgICAgICAgICk7XG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cbiAgICAgICAgcG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwgPSBuZXdEYXRhVXJsO1xuICAgICAgfVxuICAgIH1cblxuICAgIHRoaXMuaXNGaW5hbGl6ZWQgPSB0cnVlO1xuICB9XG5cbiAgLyoqXG4gICAqIOmhnuS8vOODneODvOOCuuOBruWPluW+l1xuICAgKiBAcGFyYW0gcmVzdWx0cyBNZWRpYVBpcGUgSG9saXN0aWMg44Gr44KI44KL44Od44O844K644Gu5qSc5Ye657WQ5p6cXG4gICAqIEBwYXJhbSB0aHJlc2hvbGQg44GX44GN44GE5YCkXG4gICAqIEBwYXJhbSB0YXJnZXRSYW5nZSDjg53jg7zjgrrjgpLmr5TovIPjgZnjgovnr4Tlm7IgKGFsbDog5YWo44GmLCBib2R5UG9zZTog6Lqr5L2T44Gu44G/LCBoYW5kUG9zZTog5omL5oyH44Gu44G/KVxuICAgKiBAcmV0dXJucyDpoZ7kvLzjg53jg7zjgrrjga7phY3liJdcbiAgICovXG4gIGdldFNpbWlsYXJQb3NlcyhcbiAgICByZXN1bHRzOiBSZXN1bHRzLFxuICAgIHRocmVzaG9sZDogbnVtYmVyID0gMC45LFxuICAgIHRhcmdldFJhbmdlOiAnYWxsJyB8ICdib2R5UG9zZScgfCAnaGFuZFBvc2UnID0gJ2FsbCdcbiAgKTogU2ltaWxhclBvc2VJdGVtW10ge1xuICAgIC8vIOi6q+S9k+OBruODmeOCr+ODiOODq+OCkuWPluW+l1xuICAgIGxldCBib2R5VmVjdG9yOiBCb2R5VmVjdG9yO1xuICAgIHRyeSB7XG4gICAgICBib2R5VmVjdG9yID0gUG9zZVNldC5nZXRCb2R5VmVjdG9yKChyZXN1bHRzIGFzIGFueSkuZWEpO1xuICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgIGNvbnNvbGUuZXJyb3IoYFtQb3NlU2V0XSBnZXRTaW1pbGFyUG9zZXMgLSBFcnJvciBvY2N1cnJlZGAsIGUsIHJlc3VsdHMpO1xuICAgICAgcmV0dXJuIFtdO1xuICAgIH1cbiAgICBpZiAoIWJvZHlWZWN0b3IpIHtcbiAgICAgIHRocm93ICdDb3VsZCBub3QgZ2V0IHRoZSBib2R5IHZlY3Rvcic7XG4gICAgfVxuXG4gICAgLy8g5omL5oyH44Gu44OZ44Kv44OI44Or44KS5Y+W5b6XXG4gICAgbGV0IGhhbmRWZWN0b3I6IEhhbmRWZWN0b3I7XG4gICAgaWYgKHRhcmdldFJhbmdlID09PSAnYWxsJyB8fCB0YXJnZXRSYW5nZSA9PT0gJ2hhbmRQb3NlJykge1xuICAgICAgaGFuZFZlY3RvciA9IFBvc2VTZXQuZ2V0SGFuZFZlY3RvcihcbiAgICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyxcbiAgICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3NcbiAgICAgICk7XG4gICAgICBpZiAodGFyZ2V0UmFuZ2UgPT09ICdoYW5kUG9zZScgJiYgIWhhbmRWZWN0b3IpIHtcbiAgICAgICAgdGhyb3cgJ0NvdWxkIG5vdCBnZXQgdGhlIGhhbmQgdmVjdG9yJztcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyDlkITjg53jg7zjgrrjgajjg5njgq/jg4jjg6vjgpLmr5TovINcbiAgICBjb25zdCBwb3NlcyA9IFtdO1xuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBpZiAoXG4gICAgICAgICh0YXJnZXRSYW5nZSA9PT0gJ2FsbCcgfHwgdGFyZ2V0UmFuZ2UgPT09ICdib2R5UG9zZScpICYmXG4gICAgICAgICFwb3NlLmJvZHlWZWN0b3JcbiAgICAgICkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH0gZWxzZSBpZiAodGFyZ2V0UmFuZ2UgPT09ICdoYW5kUG9zZScgJiYgIXBvc2UuaGFuZFZlY3Rvcikge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cblxuICAgICAgLypjb25zb2xlLmRlYnVnKFxuICAgICAgICAnW1Bvc2VTZXRdIGdldFNpbWlsYXJQb3NlcyAtICcsXG4gICAgICAgIHRoaXMuZ2V0VmlkZW9OYW1lKCksXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzXG4gICAgICApOyovXG5cbiAgICAgIC8vIOi6q+S9k+OBruODneODvOOCuuOBrumhnuS8vOW6puOCkuWPluW+l1xuICAgICAgbGV0IGJvZHlTaW1pbGFyaXR5OiBudW1iZXI7XG4gICAgICBpZiAoYm9keVZlY3RvciAmJiBwb3NlLmJvZHlWZWN0b3IpIHtcbiAgICAgICAgYm9keVNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEJvZHlQb3NlU2ltaWxhcml0eShcbiAgICAgICAgICBwb3NlLmJvZHlWZWN0b3IsXG4gICAgICAgICAgYm9keVZlY3RvclxuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICAvLyDmiYvmjIfjga7jg53jg7zjgrrjga7poZ7kvLzluqbjgpLlj5blvpdcbiAgICAgIGxldCBoYW5kU2ltaWxhcml0eTogbnVtYmVyO1xuICAgICAgaWYgKGhhbmRWZWN0b3IgJiYgcG9zZS5oYW5kVmVjdG9yKSB7XG4gICAgICAgIGhhbmRTaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRIYW5kU2ltaWxhcml0eShwb3NlLmhhbmRWZWN0b3IsIGhhbmRWZWN0b3IpO1xuICAgICAgfVxuXG4gICAgICAvLyDliKTlrppcbiAgICAgIGxldCBzaW1pbGFyaXR5OiBudW1iZXIsXG4gICAgICAgIGlzU2ltaWxhciA9IGZhbHNlO1xuICAgICAgaWYgKHRhcmdldFJhbmdlID09PSAnYWxsJykge1xuICAgICAgICBzaW1pbGFyaXR5ID0gTWF0aC5tYXgoYm9keVNpbWlsYXJpdHkgPz8gMCwgaGFuZFNpbWlsYXJpdHkgPz8gMCk7XG4gICAgICAgIGlmICh0aHJlc2hvbGQgPD0gYm9keVNpbWlsYXJpdHkgfHwgdGhyZXNob2xkIDw9IGhhbmRTaW1pbGFyaXR5KSB7XG4gICAgICAgICAgaXNTaW1pbGFyID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2JvZHlQb3NlJykge1xuICAgICAgICBzaW1pbGFyaXR5ID0gYm9keVNpbWlsYXJpdHk7XG4gICAgICAgIGlmICh0aHJlc2hvbGQgPD0gYm9keVNpbWlsYXJpdHkpIHtcbiAgICAgICAgICBpc1NpbWlsYXIgPSB0cnVlO1xuICAgICAgICB9XG4gICAgICB9IGVsc2UgaWYgKHRhcmdldFJhbmdlID09PSAnaGFuZFBvc2UnKSB7XG4gICAgICAgIHNpbWlsYXJpdHkgPSBoYW5kU2ltaWxhcml0eTtcbiAgICAgICAgaWYgKHRocmVzaG9sZCA8PSBoYW5kU2ltaWxhcml0eSkge1xuICAgICAgICAgIGlzU2ltaWxhciA9IHRydWU7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgaWYgKCFpc1NpbWlsYXIpIGNvbnRpbnVlO1xuXG4gICAgICAvLyDntZDmnpzjgbjov73liqBcbiAgICAgIHBvc2VzLnB1c2goe1xuICAgICAgICAuLi5wb3NlLFxuICAgICAgICBzaW1pbGFyaXR5OiBzaW1pbGFyaXR5LFxuICAgICAgICBib2R5UG9zZVNpbWlsYXJpdHk6IGJvZHlTaW1pbGFyaXR5LFxuICAgICAgICBoYW5kUG9zZVNpbWlsYXJpdHk6IGhhbmRTaW1pbGFyaXR5LFxuICAgICAgfSBhcyBTaW1pbGFyUG9zZUl0ZW0pO1xuICAgIH1cblxuICAgIHJldHVybiBwb3NlcztcbiAgfVxuXG4gIC8qKlxuICAgKiDouqvkvZPjga7lp7/li6LjgpLooajjgZnjg5njgq/jg4jjg6vjga7lj5blvpdcbiAgICogQHBhcmFtIHBvc2VMYW5kbWFya3MgTWVkaWFQaXBlIEhvbGlzdGljIOOBp+WPluW+l+OBp+OBjeOBn+i6q+S9k+OBruODr+ODvOODq+ODieW6p+aomSAocmEg6YWN5YiXKVxuICAgKiBAcmV0dXJucyDjg5njgq/jg4jjg6tcbiAgICovXG4gIHN0YXRpYyBnZXRCb2R5VmVjdG9yKFxuICAgIHBvc2VMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W11cbiAgKTogQm9keVZlY3RvciB8IHVuZGVmaW5lZCB7XG4gICAgcmV0dXJuIHtcbiAgICAgIHJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICByaWdodEVsYm93VG9SaWdodFNob3VsZGVyOiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnosXG4gICAgICBdLFxuICAgICAgbGVmdFdyaXN0VG9MZWZ0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS56LFxuICAgICAgXSxcbiAgICB9O1xuICB9XG5cbiAgLyoqXG4gICAqIOaJi+aMh+OBruWnv+WLouOCkuihqOOBmeODmeOCr+ODiOODq+OBruWPluW+l1xuICAgKiBAcGFyYW0gbGVmdEhhbmRMYW5kbWFya3MgTWVkaWFQaXBlIEhvbGlzdGljIOOBp+WPluW+l+OBp+OBjeOBn+W3puaJi+OBruato+imj+WMluW6p+aomVxuICAgKiBAcGFyYW0gcmlnaHRIYW5kTGFuZG1hcmtzIE1lZGlhUGlwZSBIb2xpc3RpYyDjgaflj5blvpfjgafjgY3jgZ/lj7PmiYvjga7mraPopo/ljJbluqfmqJlcbiAgICogQHJldHVybnMg44OZ44Kv44OI44OrXG4gICAqL1xuICBzdGF0aWMgZ2V0SGFuZFZlY3RvcihcbiAgICBsZWZ0SGFuZExhbmRtYXJrczogeyB4OiBudW1iZXI7IHk6IG51bWJlcjsgejogbnVtYmVyIH1bXSxcbiAgICByaWdodEhhbmRMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W11cbiAgKTogSGFuZFZlY3RvciB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKFxuICAgICAgKHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDApICYmXG4gICAgICAobGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDApXG4gICAgKSB7XG4gICAgICByZXR1cm4gdW5kZWZpbmVkO1xuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICAvLyDlj7PmiYsgLSDopqrmjIdcbiAgICAgIHJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnggLSByaWdodEhhbmRMYW5kbWFya3NbM10ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbM10ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnogLSByaWdodEhhbmRMYW5kbWFya3NbM10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgIHJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnggLSByaWdodEhhbmRMYW5kbWFya3NbN10ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbN10ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnogLSByaWdodEhhbmRMYW5kbWFya3NbN10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnggLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnkgLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnogLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDkuK3mjIdcbiAgICAgIHJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMl0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzEyXS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzExXS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTJdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTFdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTFdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTBdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMF0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzExXS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzEwXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOiWrOaMh1xuICAgICAgcmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNl0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE2XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTZdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTVdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE0XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTVdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTRdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDlsI/mjIdcbiAgICAgIHJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzIwXS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMjBdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTldLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1syMF0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE4XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTldLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMThdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAgIGxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbNF0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzNdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzRdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1szXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s0XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbM10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbM10ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzJdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1syXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1szXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgIGxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbOF0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzddLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzhdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1s3XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s4XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbN10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbN10ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzZdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzddLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1s2XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s3XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbNl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDkuK3mjIdcbiAgICAgIGxlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOiWrOaMh1xuICAgICAgbGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDlsI/mjIdcbiAgICAgIGxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgfTtcbiAgfVxuXG4gIC8qKlxuICAgKiBCb2R5VmVjdG9yIOmWk+OBjOmhnuS8vOOBl+OBpuOBhOOCi+OBi+OBqeOBhuOBi+OBruWIpOWumlxuICAgKiBAcGFyYW0gYm9keVZlY3RvckEg5q+U6LyD5YWI44GuIEJvZHlWZWN0b3JcbiAgICogQHBhcmFtIGJvZHlWZWN0b3JCIOavlOi8g+WFg+OBriBCb2R5VmVjdG9yXG4gICAqIEBwYXJhbSB0aHJlc2hvbGQg44GX44GN44GE5YCkXG4gICAqIEByZXR1cm5zIOmhnuS8vOOBl+OBpuOBhOOCi+OBi+OBqeOBhuOBi1xuICAgKi9cbiAgc3RhdGljIGlzU2ltaWxhckJvZHlQb3NlKFxuICAgIGJvZHlWZWN0b3JBOiBCb2R5VmVjdG9yLFxuICAgIGJvZHlWZWN0b3JCOiBCb2R5VmVjdG9yLFxuICAgIHRocmVzaG9sZCA9IDAuOFxuICApOiBib29sZWFuIHtcbiAgICBsZXQgaXNTaW1pbGFyID0gZmFsc2U7XG4gICAgY29uc3Qgc2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0Qm9keVBvc2VTaW1pbGFyaXR5KGJvZHlWZWN0b3JBLCBib2R5VmVjdG9yQik7XG4gICAgaWYgKHNpbWlsYXJpdHkgPj0gdGhyZXNob2xkKSBpc1NpbWlsYXIgPSB0cnVlO1xuXG4gICAgLy8gY29uc29sZS5kZWJ1ZyhgW1Bvc2VTZXRdIGlzU2ltaWxhclBvc2VgLCBpc1NpbWlsYXIsIHNpbWlsYXJpdHkpO1xuXG4gICAgcmV0dXJuIGlzU2ltaWxhcjtcbiAgfVxuXG4gIC8qKlxuICAgKiDouqvkvZPjg53jg7zjgrrjga7poZ7kvLzluqbjga7lj5blvpdcbiAgICogQHBhcmFtIGJvZHlWZWN0b3JBIOavlOi8g+WFiOOBriBCb2R5VmVjdG9yXG4gICAqIEBwYXJhbSBib2R5VmVjdG9yQiDmr5TovIPlhYPjga4gQm9keVZlY3RvclxuICAgKiBAcmV0dXJucyDpoZ7kvLzluqZcbiAgICovXG4gIHN0YXRpYyBnZXRCb2R5UG9zZVNpbWlsYXJpdHkoXG4gICAgYm9keVZlY3RvckE6IEJvZHlWZWN0b3IsXG4gICAgYm9keVZlY3RvckI6IEJvZHlWZWN0b3JcbiAgKTogbnVtYmVyIHtcbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXMgPSB7XG4gICAgICBsZWZ0V3Jpc3RUb0xlZnRFbGJvdzogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5sZWZ0V3Jpc3RUb0xlZnRFbGJvdyxcbiAgICAgICAgYm9keVZlY3RvckIubGVmdFdyaXN0VG9MZWZ0RWxib3dcbiAgICAgICksXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5sZWZ0RWxib3dUb0xlZnRTaG91bGRlcixcbiAgICAgICAgYm9keVZlY3RvckIubGVmdEVsYm93VG9MZWZ0U2hvdWxkZXJcbiAgICAgICksXG4gICAgICByaWdodFdyaXN0VG9SaWdodEVsYm93OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLnJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3csXG4gICAgICAgIGJvZHlWZWN0b3JCLnJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3dcbiAgICAgICksXG4gICAgICByaWdodEVsYm93VG9SaWdodFNob3VsZGVyOiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXIsXG4gICAgICAgIGJvZHlWZWN0b3JCLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXJcbiAgICAgICksXG4gICAgfTtcblxuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllc1N1bSA9IE9iamVjdC52YWx1ZXMoY29zU2ltaWxhcml0aWVzKS5yZWR1Y2UoXG4gICAgICAoc3VtLCB2YWx1ZSkgPT4gc3VtICsgdmFsdWUsXG4gICAgICAwXG4gICAgKTtcbiAgICByZXR1cm4gY29zU2ltaWxhcml0aWVzU3VtIC8gT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzKS5sZW5ndGg7XG4gIH1cblxuICAvKipcbiAgICogSGFuZFZlY3RvciDplpPjgYzpoZ7kvLzjgZfjgabjgYTjgovjgYvjganjgYbjgYvjga7liKTlrppcbiAgICogQHBhcmFtIGhhbmRWZWN0b3JBIOavlOi8g+WFiOOBriBIYW5kVmVjdG9yXG4gICAqIEBwYXJhbSBoYW5kVmVjdG9yQiDmr5TovIPlhYPjga4gSGFuZFZlY3RvclxuICAgKiBAcGFyYW0gdGhyZXNob2xkIOOBl+OBjeOBhOWApFxuICAgKiBAcmV0dXJucyDpoZ7kvLzjgZfjgabjgYTjgovjgYvjganjgYbjgYtcbiAgICovXG4gIHN0YXRpYyBpc1NpbWlsYXJIYW5kUG9zZShcbiAgICBoYW5kVmVjdG9yQTogSGFuZFZlY3RvcixcbiAgICBoYW5kVmVjdG9yQjogSGFuZFZlY3RvcixcbiAgICB0aHJlc2hvbGQgPSAwLjc1XG4gICk6IGJvb2xlYW4ge1xuICAgIGNvbnN0IHNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEhhbmRTaW1pbGFyaXR5KGhhbmRWZWN0b3JBLCBoYW5kVmVjdG9yQik7XG4gICAgaWYgKHNpbWlsYXJpdHkgPT09IC0xKSB7XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG4gICAgcmV0dXJuIHNpbWlsYXJpdHkgPj0gdGhyZXNob2xkO1xuICB9XG5cbiAgLyoqXG4gICAqIOaJi+OBruODneODvOOCuuOBrumhnuS8vOW6puOBruWPluW+l1xuICAgKiBAcGFyYW0gaGFuZFZlY3RvckEg5q+U6LyD5YWI44GuIEhhbmRWZWN0b3JcbiAgICogQHBhcmFtIGhhbmRWZWN0b3JCIOavlOi8g+WFg+OBriBIYW5kVmVjdG9yXG4gICAqIEByZXR1cm5zIOmhnuS8vOW6plxuICAgKi9cbiAgc3RhdGljIGdldEhhbmRTaW1pbGFyaXR5KFxuICAgIGhhbmRWZWN0b3JBOiBIYW5kVmVjdG9yLFxuICAgIGhhbmRWZWN0b3JCOiBIYW5kVmVjdG9yXG4gICk6IG51bWJlciB7XG4gICAgY29uc3QgY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kID1cbiAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbCB8fFxuICAgICAgaGFuZFZlY3RvckIucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ID09PSBudWxsXG4gICAgICAgID8gdW5kZWZpbmVkXG4gICAgICAgIDoge1xuICAgICAgICAgICAgLy8g5Y+z5omLIC0g6Kaq5oyHXG4gICAgICAgICAgICByaWdodFRodW1iVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgICAgICAgIHJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOS4reaMh1xuICAgICAgICAgICAgcmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5Y+z5omLIC0g6Jas5oyHXG4gICAgICAgICAgICByaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICAgICAgICAgcmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgIH07XG5cbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCA9XG4gICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbCB8fFxuICAgICAgaGFuZFZlY3RvckIubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICAgPyB1bmRlZmluZWRcbiAgICAgICAgOiB7XG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAgICAgICAgIGxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIGxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOS6uuW3ruOBl+aMh1xuICAgICAgICAgICAgbGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5Lit5oyHXG4gICAgICAgICAgICBsZWZ0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDolqzmjIdcbiAgICAgICAgICAgIGxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIGxlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOWwj+aMh1xuICAgICAgICAgICAgbGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgIH07XG5cbiAgICAvLyDlt6bmiYvjga7poZ7kvLzluqZcbiAgICBsZXQgY29zU2ltaWxhcml0aWVzU3VtTGVmdEhhbmQgPSAwO1xuICAgIGlmIChjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCkge1xuICAgICAgY29zU2ltaWxhcml0aWVzU3VtTGVmdEhhbmQgPSBPYmplY3QudmFsdWVzKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZFxuICAgICAgKS5yZWR1Y2UoKHN1bSwgdmFsdWUpID0+IHN1bSArIHZhbHVlLCAwKTtcbiAgICB9XG5cbiAgICAvLyDlj7PmiYvjga7poZ7kvLzluqZcbiAgICBsZXQgY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kID0gMDtcbiAgICBpZiAoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kKSB7XG4gICAgICBjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgPSBPYmplY3QudmFsdWVzKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmRcbiAgICAgICkucmVkdWNlKChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSwgMCk7XG4gICAgfVxuXG4gICAgLy8g5ZCI566X44GV44KM44Gf6aGe5Ly85bqmXG4gICAgaWYgKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCAmJiBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCkge1xuICAgICAgcmV0dXJuIChcbiAgICAgICAgKGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCArIGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kKSAvXG4gICAgICAgIChPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQhKS5sZW5ndGggK1xuICAgICAgICAgIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kISkubGVuZ3RoKVxuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCkge1xuICAgICAgaWYgKFxuICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCAhPT0gbnVsbCAmJlxuICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbFxuICAgICAgKSB7XG4gICAgICAgIC8vIGhhbmRWZWN0b3JCIOOBp+W3puaJi+OBjOOBguOCi+OBruOBqyBoYW5kVmVjdG9yQSDjgaflt6bmiYvjgYzjgarjgYTloLTlkIjjgIHpoZ7kvLzluqbjgpLmuJvjgonjgZlcbiAgICAgICAgY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgICBgW1Bvc2VTZXRdIGdldEhhbmRTaW1pbGFyaXR5IC0gQWRqdXN0IHNpbWlsYXJpdHksIGJlY2F1c2UgbGVmdCBoYW5kIG5vdCBmb3VuZC4uLmBcbiAgICAgICAgKTtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICBjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgL1xuICAgICAgICAgIChPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQhKS5sZW5ndGggKiAyKVxuICAgICAgICApO1xuICAgICAgfVxuICAgICAgcmV0dXJuIChcbiAgICAgICAgY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kIC9cbiAgICAgICAgT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kISkubGVuZ3RoXG4gICAgICApO1xuICAgIH0gZWxzZSBpZiAoY29zU2ltaWxhcml0aWVzTGVmdEhhbmQpIHtcbiAgICAgIGlmIChcbiAgICAgICAgaGFuZFZlY3RvckIucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ICE9PSBudWxsICYmXG4gICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbFxuICAgICAgKSB7XG4gICAgICAgIC8vIGhhbmRWZWN0b3JCIOOBp+WPs+aJi+OBjOOBguOCi+OBruOBqyBoYW5kVmVjdG9yQSDjgaflj7PmiYvjgYzjgarjgYTloLTlkIjjgIHpoZ7kvLzluqbjgpLmuJvjgonjgZlcbiAgICAgICAgY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgICBgW1Bvc2VTZXRdIGdldEhhbmRTaW1pbGFyaXR5IC0gQWRqdXN0IHNpbWlsYXJpdHksIGJlY2F1c2UgcmlnaHQgaGFuZCBub3QgZm91bmQuLi5gXG4gICAgICAgICk7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgY29zU2ltaWxhcml0aWVzU3VtTGVmdEhhbmQgL1xuICAgICAgICAgIChPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCEpLmxlbmd0aCAqIDIpXG4gICAgICAgICk7XG4gICAgICB9XG4gICAgICByZXR1cm4gKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCAvXG4gICAgICAgIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kISkubGVuZ3RoXG4gICAgICApO1xuICAgIH1cblxuICAgIHJldHVybiAtMTtcbiAgfVxuXG4gIC8qKlxuICAgKiBaSVAg44OV44Kh44Kk44Or44Go44GX44Gm44Gu44K344Oq44Ki44Op44Kk44K6XG4gICAqIEByZXR1cm5zIFpJUOODleOCoeOCpOODqyAoQmxvYiDlvaLlvI8pXG4gICAqL1xuICBwdWJsaWMgYXN5bmMgZ2V0WmlwKCk6IFByb21pc2U8QmxvYj4ge1xuICAgIGNvbnN0IGpzWmlwID0gbmV3IEpTWmlwKCk7XG4gICAganNaaXAuZmlsZSgncG9zZXMuanNvbicsIGF3YWl0IHRoaXMuZ2V0SnNvbigpKTtcblxuICAgIGNvbnN0IGltYWdlRmlsZUV4dCA9IHRoaXMuZ2V0RmlsZUV4dGVuc2lvbkJ5TWltZSh0aGlzLklNQUdFX01JTUUpO1xuXG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGlmIChwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgY29uc3QgaW5kZXggPVxuICAgICAgICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UuZnJhbWVJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcbiAgICAgICAgICBqc1ppcC5maWxlKGBmcmFtZS0ke3Bvc2UuaWR9LiR7aW1hZ2VGaWxlRXh0fWAsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZyYW1lIGltYWdlYCxcbiAgICAgICAgICAgIGVycm9yXG4gICAgICAgICAgKTtcbiAgICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgaWYgKHBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UucG9zZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYHBvc2UtJHtwb3NlLmlkfS4ke2ltYWdlRmlsZUV4dH1gLCBiYXNlNjQsIHtcbiAgICAgICAgICAgIGJhc2U2NDogdHJ1ZSxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgW1Bvc2VFeHBvcnRlclNlcnZpY2VdIHB1c2ggLSBDb3VsZCBub3QgcHVzaCBmcmFtZSBpbWFnZWAsXG4gICAgICAgICAgICBlcnJvclxuICAgICAgICAgICk7XG4gICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGlmIChwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcbiAgICAgICAgICBqc1ppcC5maWxlKGBmYWNlLSR7cG9zZS5pZH0uJHtpbWFnZUZpbGVFeHR9YCwgYmFzZTY0LCB7XG4gICAgICAgICAgICBiYXNlNjQ6IHRydWUsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlRXhwb3J0ZXJTZXJ2aWNlXSBwdXNoIC0gQ291bGQgbm90IHB1c2ggZmFjZSBmcmFtZSBpbWFnZWAsXG4gICAgICAgICAgICBlcnJvclxuICAgICAgICAgICk7XG4gICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gYXdhaXQganNaaXAuZ2VuZXJhdGVBc3luYyh7IHR5cGU6ICdibG9iJyB9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBKU09OIOaWh+Wtl+WIl+OBqOOBl+OBpuOBruOCt+ODquOCouODqeOCpOOCulxuICAgKiBAcmV0dXJucyBKU09OIOaWh+Wtl+WIl1xuICAgKi9cbiAgcHVibGljIGFzeW5jIGdldEpzb24oKTogUHJvbWlzZTxzdHJpbmc+IHtcbiAgICBpZiAodGhpcy52aWRlb01ldGFkYXRhID09PSB1bmRlZmluZWQgfHwgdGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKVxuICAgICAgcmV0dXJuICd7fSc7XG5cbiAgICBpZiAoIXRoaXMuaXNGaW5hbGl6ZWQpIHtcbiAgICAgIGF3YWl0IHRoaXMuZmluYWxpemUoKTtcbiAgICB9XG5cbiAgICBsZXQgcG9zZUxhbmRtYXJrTWFwcGluZ3MgPSBbXTtcbiAgICBmb3IgKGNvbnN0IGtleSBvZiBPYmplY3Qua2V5cyhQT1NFX0xBTkRNQVJLUykpIHtcbiAgICAgIGNvbnN0IGluZGV4OiBudW1iZXIgPSBQT1NFX0xBTkRNQVJLU1trZXkgYXMga2V5b2YgdHlwZW9mIFBPU0VfTEFORE1BUktTXTtcbiAgICAgIHBvc2VMYW5kbWFya01hcHBpbmdzW2luZGV4XSA9IGtleTtcbiAgICB9XG5cbiAgICBjb25zdCBqc29uOiBQb3NlU2V0SnNvbiA9IHtcbiAgICAgIGdlbmVyYXRvcjogJ21wLXZpZGVvLXBvc2UtZXh0cmFjdG9yJyxcbiAgICAgIHZlcnNpb246IDEsXG4gICAgICB2aWRlbzogdGhpcy52aWRlb01ldGFkYXRhISxcbiAgICAgIHBvc2VzOiB0aGlzLnBvc2VzLm1hcCgocG9zZTogUG9zZVNldEl0ZW0pOiBQb3NlU2V0SnNvbkl0ZW0gPT4ge1xuICAgICAgICAvLyBCb2R5VmVjdG9yIOOBruWcp+e4rlxuICAgICAgICBjb25zdCBib2R5VmVjdG9yID0gW107XG4gICAgICAgIGZvciAoY29uc3Qga2V5IG9mIFBvc2VTZXQuQk9EWV9WRUNUT1JfTUFQUElOR1MpIHtcbiAgICAgICAgICBib2R5VmVjdG9yLnB1c2gocG9zZS5ib2R5VmVjdG9yW2tleSBhcyBrZXlvZiBCb2R5VmVjdG9yXSk7XG4gICAgICAgIH1cblxuICAgICAgICAvLyBIYW5kVmVjdG9yIOOBruWcp+e4rlxuICAgICAgICBsZXQgaGFuZFZlY3RvcjogKG51bWJlcltdIHwgbnVsbClbXSB8IHVuZGVmaW5lZCA9IHVuZGVmaW5lZDtcbiAgICAgICAgaWYgKHBvc2UuaGFuZFZlY3Rvcikge1xuICAgICAgICAgIGhhbmRWZWN0b3IgPSBbXTtcbiAgICAgICAgICBmb3IgKGNvbnN0IGtleSBvZiBQb3NlU2V0LkhBTkRfVkVDVE9SX01BUFBJTkdTKSB7XG4gICAgICAgICAgICBoYW5kVmVjdG9yLnB1c2gocG9zZS5oYW5kVmVjdG9yW2tleSBhcyBrZXlvZiBIYW5kVmVjdG9yXSk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG5cbiAgICAgICAgLy8gUG9zZVNldEpzb25JdGVtIOOBriBwb3NlIOOCquODluOCuOOCp+OCr+ODiOOCkueUn+aIkFxuICAgICAgICByZXR1cm4ge1xuICAgICAgICAgIGlkOiBwb3NlLmlkLFxuICAgICAgICAgIHQ6IHBvc2UudGltZU1pbGlzZWNvbmRzLFxuICAgICAgICAgIGQ6IHBvc2UuZHVyYXRpb25NaWxpc2Vjb25kcyxcbiAgICAgICAgICBwOiBwb3NlLnBvc2UsXG4gICAgICAgICAgbDogcG9zZS5sZWZ0SGFuZCxcbiAgICAgICAgICByOiBwb3NlLnJpZ2h0SGFuZCxcbiAgICAgICAgICB2OiBib2R5VmVjdG9yLFxuICAgICAgICAgIGg6IGhhbmRWZWN0b3IsXG4gICAgICAgICAgZTogcG9zZS5leHRlbmRlZERhdGEsXG4gICAgICAgICAgbWQ6IHBvc2UubWVyZ2VkRHVyYXRpb25NaWxpc2Vjb25kcyxcbiAgICAgICAgICBtdDogcG9zZS5tZXJnZWRUaW1lTWlsaXNlY29uZHMsXG4gICAgICAgIH07XG4gICAgICB9KSxcbiAgICAgIHBvc2VMYW5kbWFya01hcHBwaW5nczogcG9zZUxhbmRtYXJrTWFwcGluZ3MsXG4gICAgfTtcblxuICAgIHJldHVybiBKU09OLnN0cmluZ2lmeShqc29uKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBKU09OIOOBi+OCieOBruiqreOBv+i+vOOBv1xuICAgKiBAcGFyYW0ganNvbiBKU09OIOaWh+Wtl+WIlyDjgb7jgZ/jga8gSlNPTiDjgqrjg5bjgrjjgqfjgq/jg4hcbiAgICovXG4gIGxvYWRKc29uKGpzb246IHN0cmluZyB8IGFueSkge1xuICAgIGNvbnN0IHBhcnNlZEpzb24gPSB0eXBlb2YganNvbiA9PT0gJ3N0cmluZycgPyBKU09OLnBhcnNlKGpzb24pIDoganNvbjtcblxuICAgIGlmIChwYXJzZWRKc29uLmdlbmVyYXRvciAhPT0gJ21wLXZpZGVvLXBvc2UtZXh0cmFjdG9yJykge1xuICAgICAgdGhyb3cgJ+S4jeato+OBquODleOCoeOCpOODqyc7XG4gICAgfSBlbHNlIGlmIChwYXJzZWRKc29uLnZlcnNpb24gIT09IDEpIHtcbiAgICAgIHRocm93ICfmnKrlr77lv5zjga7jg5Djg7zjgrjjg6fjg7MnO1xuICAgIH1cblxuICAgIHRoaXMudmlkZW9NZXRhZGF0YSA9IHBhcnNlZEpzb24udmlkZW87XG4gICAgdGhpcy5wb3NlcyA9IHBhcnNlZEpzb24ucG9zZXMubWFwKChpdGVtOiBQb3NlU2V0SnNvbkl0ZW0pOiBQb3NlU2V0SXRlbSA9PiB7XG4gICAgICBjb25zdCBib2R5VmVjdG9yOiBhbnkgPSB7fTtcbiAgICAgIFBvc2VTZXQuQk9EWV9WRUNUT1JfTUFQUElOR1MubWFwKChrZXksIGluZGV4KSA9PiB7XG4gICAgICAgIGJvZHlWZWN0b3Jba2V5IGFzIGtleW9mIEJvZHlWZWN0b3JdID0gaXRlbS52W2luZGV4XTtcbiAgICAgIH0pO1xuXG4gICAgICBjb25zdCBoYW5kVmVjdG9yOiBhbnkgPSB7fTtcbiAgICAgIGlmIChpdGVtLmgpIHtcbiAgICAgICAgUG9zZVNldC5IQU5EX1ZFQ1RPUl9NQVBQSU5HUy5tYXAoKGtleSwgaW5kZXgpID0+IHtcbiAgICAgICAgICBoYW5kVmVjdG9yW2tleSBhcyBrZXlvZiBIYW5kVmVjdG9yXSA9IGl0ZW0uaCFbaW5kZXhdO1xuICAgICAgICB9KTtcbiAgICAgIH1cblxuICAgICAgcmV0dXJuIHtcbiAgICAgICAgaWQ6XG4gICAgICAgICAgaXRlbS5pZCA9PT0gdW5kZWZpbmVkXG4gICAgICAgICAgICA/IFBvc2VTZXQuZ2V0SWRCeVRpbWVNaWxpc2Vjb25kcyhpdGVtLnQpXG4gICAgICAgICAgICA6IGl0ZW0uaWQsXG4gICAgICAgIHRpbWVNaWxpc2Vjb25kczogaXRlbS50LFxuICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBpdGVtLmQsXG4gICAgICAgIHBvc2U6IGl0ZW0ucCxcbiAgICAgICAgbGVmdEhhbmQ6IGl0ZW0ubCxcbiAgICAgICAgcmlnaHRIYW5kOiBpdGVtLnIsXG4gICAgICAgIGJvZHlWZWN0b3I6IGJvZHlWZWN0b3IsXG4gICAgICAgIGhhbmRWZWN0b3I6IGhhbmRWZWN0b3IsXG4gICAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiB1bmRlZmluZWQsXG4gICAgICAgIGV4dGVuZGVkRGF0YTogaXRlbS5lLFxuICAgICAgICBkZWJ1ZzogdW5kZWZpbmVkLFxuICAgICAgICBtZXJnZWREdXJhdGlvbk1pbGlzZWNvbmRzOiBpdGVtLm1kLFxuICAgICAgICBtZXJnZWRUaW1lTWlsaXNlY29uZHM6IGl0ZW0ubXQsXG4gICAgICB9O1xuICAgIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIFpJUCDjg5XjgqHjgqTjg6vjgYvjgonjga7oqq3jgb/ovrzjgb9cbiAgICogQHBhcmFtIGJ1ZmZlciBaSVAg44OV44Kh44Kk44Or44GuIEJ1ZmZlclxuICAgKiBAcGFyYW0gaW5jbHVkZUltYWdlcyDnlLvlg4/jgpLlsZXplovjgZnjgovjgYvjganjgYbjgYtcbiAgICovXG4gIGFzeW5jIGxvYWRaaXAoYnVmZmVyOiBBcnJheUJ1ZmZlciwgaW5jbHVkZUltYWdlczogYm9vbGVhbiA9IHRydWUpIHtcbiAgICBjb25zdCBqc1ppcCA9IG5ldyBKU1ppcCgpO1xuICAgIGNvbnNvbGUuZGVidWcoYFtQb3NlU2V0XSBpbml0Li4uYCk7XG4gICAgY29uc3QgemlwID0gYXdhaXQganNaaXAubG9hZEFzeW5jKGJ1ZmZlciwgeyBiYXNlNjQ6IGZhbHNlIH0pO1xuICAgIGlmICghemlwKSB0aHJvdyAnWklQ44OV44Kh44Kk44Or44KS6Kqt44G/6L6844KB44G+44Gb44KT44Gn44GX44GfJztcblxuICAgIGNvbnN0IGpzb24gPSBhd2FpdCB6aXAuZmlsZSgncG9zZXMuanNvbicpPy5hc3luYygndGV4dCcpO1xuICAgIGlmIChqc29uID09PSB1bmRlZmluZWQpIHtcbiAgICAgIHRocm93ICdaSVDjg5XjgqHjgqTjg6vjgasgcG9zZS5qc29uIOOBjOWQq+OBvuOCjOOBpuOBhOOBvuOBm+OCkyc7XG4gICAgfVxuXG4gICAgdGhpcy5sb2FkSnNvbihqc29uKTtcblxuICAgIGNvbnN0IGZpbGVFeHQgPSB0aGlzLmdldEZpbGVFeHRlbnNpb25CeU1pbWUodGhpcy5JTUFHRV9NSU1FKTtcblxuICAgIGlmIChpbmNsdWRlSW1hZ2VzKSB7XG4gICAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgICBsZXQgaW1hZ2VCYXNlNjQ6IHN0cmluZztcbiAgICAgICAgICBpZiAoemlwLmZpbGUoYGZyYW1lLSR7cG9zZS5pZH0uJHtmaWxlRXh0fWApKSB7XG4gICAgICAgICAgICBpbWFnZUJhc2U2NCA9IGF3YWl0IHppcFxuICAgICAgICAgICAgICAuZmlsZShgZnJhbWUtJHtwb3NlLmlkfS4ke2ZpbGVFeHR9YClcbiAgICAgICAgICAgICAgPy5hc3luYygnYmFzZTY0Jyk7XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGltYWdlQmFzZTY0ID0gYXdhaXQgemlwXG4gICAgICAgICAgICAgIC5maWxlKGBmcmFtZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ZpbGVFeHR9YClcbiAgICAgICAgICAgICAgPy5hc3luYygnYmFzZTY0Jyk7XG4gICAgICAgICAgfVxuICAgICAgICAgIGlmIChpbWFnZUJhc2U2NCkge1xuICAgICAgICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybCA9IGBkYXRhOiR7dGhpcy5JTUFHRV9NSU1FfTtiYXNlNjQsJHtpbWFnZUJhc2U2NH1gO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBpZiAoIXBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICAgIGNvbnN0IHBvc2VJbWFnZUZpbGVOYW1lID0gYHBvc2UtJHtwb3NlLmlkfS4ke2ZpbGVFeHR9YDtcbiAgICAgICAgICBjb25zdCBpbWFnZUJhc2U2NCA9IGF3YWl0IHppcFxuICAgICAgICAgICAgLmZpbGUocG9zZUltYWdlRmlsZU5hbWUpXG4gICAgICAgICAgICA/LmFzeW5jKCdiYXNlNjQnKTtcbiAgICAgICAgICBpZiAoaW1hZ2VCYXNlNjQpIHtcbiAgICAgICAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybCA9IGBkYXRhOiR7dGhpcy5JTUFHRV9NSU1FfTtiYXNlNjQsJHtpbWFnZUJhc2U2NH1gO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIHN0YXRpYyBnZXRDb3NTaW1pbGFyaXR5KGE6IG51bWJlcltdLCBiOiBudW1iZXJbXSkge1xuICAgIGlmIChjb3NTaW1pbGFyaXR5QSkge1xuICAgICAgcmV0dXJuIGNvc1NpbWlsYXJpdHlBKGEsIGIpO1xuICAgIH1cbiAgICByZXR1cm4gY29zU2ltaWxhcml0eUIoYSwgYik7XG4gIH1cblxuICBwcml2YXRlIHB1c2hQb3NlRnJvbVNpbWlsYXJQb3NlUXVldWUobmV4dFBvc2VUaW1lTWlsaXNlY29uZHM/OiBudW1iZXIpIHtcbiAgICBpZiAodGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCA9PT0gMCkgcmV0dXJuO1xuXG4gICAgaWYgKHRoaXMuc2ltaWxhclBvc2VRdWV1ZS5sZW5ndGggPT09IDEpIHtcbiAgICAgIC8vIOmhnuS8vOODneODvOOCuuOCreODpeODvOOBq+ODneODvOOCuuOBjOS4gOOBpOOBl+OBi+OBquOBhOWgtOWQiOOAgeW9k+ipsuODneODvOOCuuOCkuODneODvOOCuumFjeWIl+OBuOi/veWKoFxuICAgICAgY29uc3QgcG9zZSA9IHRoaXMuc2ltaWxhclBvc2VRdWV1ZVswXTtcbiAgICAgIHRoaXMucG9zZXMucHVzaChwb3NlKTtcbiAgICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZSA9IFtdO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIC8vIOWQhOODneODvOOCuuOBruaMgee2muaZgumWk+OCkuioreWumlxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCAtIDE7IGkrKykge1xuICAgICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlW2ldLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWVbaSArIDFdLnRpbWVNaWxpc2Vjb25kcyAtXG4gICAgICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZVtpXS50aW1lTWlsaXNlY29uZHM7XG4gICAgfVxuICAgIGlmIChuZXh0UG9zZVRpbWVNaWxpc2Vjb25kcykge1xuICAgICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlW1xuICAgICAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWUubGVuZ3RoIC0gMVxuICAgICAgXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgbmV4dFBvc2VUaW1lTWlsaXNlY29uZHMgLVxuICAgICAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWVbdGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCAtIDFdLnRpbWVNaWxpc2Vjb25kcztcbiAgICB9XG5cbiAgICAvLyDpoZ7kvLzjg53jg7zjgrrjgq3jg6Xjg7zjga7kuK3jgYvjgonmnIDjgoLmjIHntprmmYLplpPjgYzplbfjgYTjg53jg7zjgrrjgpLpgbjmip5cbiAgICBjb25zdCBzZWxlY3RlZFBvc2UgPSBQb3NlU2V0LmdldFN1aXRhYmxlUG9zZUJ5UG9zZXModGhpcy5zaW1pbGFyUG9zZVF1ZXVlKTtcblxuICAgIC8vIOmBuOaKnuOBleOCjOOBquOBi+OBo+OBn+ODneODvOOCuuOCkuWIl+aMmVxuICAgIHNlbGVjdGVkUG9zZS5kZWJ1Zy5kdXBsaWNhdGVkSXRlbXMgPSB0aGlzLnNpbWlsYXJQb3NlUXVldWVcbiAgICAgIC5maWx0ZXIoKGl0ZW06IFBvc2VTZXRJdGVtKSA9PiB7XG4gICAgICAgIHJldHVybiBpdGVtLnRpbWVNaWxpc2Vjb25kcyAhPT0gc2VsZWN0ZWRQb3NlLnRpbWVNaWxpc2Vjb25kcztcbiAgICAgIH0pXG4gICAgICAubWFwKChpdGVtOiBQb3NlU2V0SXRlbSkgPT4ge1xuICAgICAgICByZXR1cm4ge1xuICAgICAgICAgIGlkOiBpdGVtLmlkLFxuICAgICAgICAgIHRpbWVNaWxpc2Vjb25kczogaXRlbS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgICAgZHVyYXRpb25NaWxpc2Vjb25kczogaXRlbS5kdXJhdGlvbk1pbGlzZWNvbmRzLFxuICAgICAgICB9O1xuICAgICAgfSk7XG5cbiAgICAvLyDpgbjmip7jgZXjgozjgZ/jg53jg7zjgrrjga7mg4XloLHjgpLmm7TmlrBcbiAgICBzZWxlY3RlZFBvc2UubWVyZ2VkVGltZU1pbGlzZWNvbmRzID1cbiAgICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZVswXS50aW1lTWlsaXNlY29uZHM7XG4gICAgc2VsZWN0ZWRQb3NlLm1lcmdlZER1cmF0aW9uTWlsaXNlY29uZHMgPSB0aGlzLnNpbWlsYXJQb3NlUXVldWUucmVkdWNlKFxuICAgICAgKHN1bTogbnVtYmVyLCBpdGVtOiBQb3NlU2V0SXRlbSkgPT4ge1xuICAgICAgICByZXR1cm4gc3VtICsgaXRlbS5kdXJhdGlvbk1pbGlzZWNvbmRzO1xuICAgICAgfSxcbiAgICAgIDBcbiAgICApO1xuICAgIHNlbGVjdGVkUG9zZS5pZCA9IFBvc2VTZXQuZ2V0SWRCeVRpbWVNaWxpc2Vjb25kcyhcbiAgICAgIHNlbGVjdGVkUG9zZS5tZXJnZWRUaW1lTWlsaXNlY29uZHNcbiAgICApO1xuXG4gICAgLy8g5b2T6Kmy44Od44O844K644KS44Od44O844K66YWN5YiX44G46L+95YqgXG4gICAgaWYgKHRoaXMuSVNfRU5BQkxFRF9SRU1PVkVfRFVQTElDQVRFRF9QT1NFU19GT1JfQVJPVU5EKSB7XG4gICAgICB0aGlzLnBvc2VzLnB1c2goc2VsZWN0ZWRQb3NlKTtcbiAgICB9IGVsc2Uge1xuICAgICAgLy8g44OH44OQ44OD44Kw55SoXG4gICAgICB0aGlzLnBvc2VzLnB1c2goLi4udGhpcy5zaW1pbGFyUG9zZVF1ZXVlKTtcbiAgICB9XG5cbiAgICAvLyDpoZ7kvLzjg53jg7zjgrrjgq3jg6Xjg7zjgpLjgq/jg6rjgqJcbiAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWUgPSBbXTtcbiAgfVxuXG4gIHJlbW92ZUR1cGxpY2F0ZWRQb3NlcygpOiB2b2lkIHtcbiAgICAvLyDlhajjg53jg7zjgrrjgpLmr5TovIPjgZfjgabpoZ7kvLzjg53jg7zjgrrjgpLliYrpmaRcbiAgICBjb25zdCBuZXdQb3NlczogUG9zZVNldEl0ZW1bXSA9IFtdLFxuICAgICAgcmVtb3ZlZFBvc2VzOiBQb3NlU2V0SXRlbVtdID0gW107XG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGxldCBkdXBsaWNhdGVkUG9zZTogUG9zZVNldEl0ZW07XG4gICAgICBmb3IgKGNvbnN0IGluc2VydGVkUG9zZSBvZiBuZXdQb3Nlcykge1xuICAgICAgICBjb25zdCBpc1NpbWlsYXJCb2R5UG9zZSA9IFBvc2VTZXQuaXNTaW1pbGFyQm9keVBvc2UoXG4gICAgICAgICAgcG9zZS5ib2R5VmVjdG9yLFxuICAgICAgICAgIGluc2VydGVkUG9zZS5ib2R5VmVjdG9yXG4gICAgICAgICk7XG4gICAgICAgIGNvbnN0IGlzU2ltaWxhckhhbmRQb3NlID1cbiAgICAgICAgICBwb3NlLmhhbmRWZWN0b3IgJiYgaW5zZXJ0ZWRQb3NlLmhhbmRWZWN0b3JcbiAgICAgICAgICAgID8gUG9zZVNldC5pc1NpbWlsYXJIYW5kUG9zZShcbiAgICAgICAgICAgICAgICBwb3NlLmhhbmRWZWN0b3IsXG4gICAgICAgICAgICAgICAgaW5zZXJ0ZWRQb3NlLmhhbmRWZWN0b3IsXG4gICAgICAgICAgICAgICAgMC45XG4gICAgICAgICAgICAgIClcbiAgICAgICAgICAgIDogZmFsc2U7XG5cbiAgICAgICAgaWYgKGlzU2ltaWxhckJvZHlQb3NlICYmIGlzU2ltaWxhckhhbmRQb3NlKSB7XG4gICAgICAgICAgLy8g6Lqr5L2T44O75omL44Go44KC44Gr6aGe5Ly844Od44O844K644Gq44KJ44GwXG4gICAgICAgICAgZHVwbGljYXRlZFBvc2UgPSBpbnNlcnRlZFBvc2U7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgaWYgKGR1cGxpY2F0ZWRQb3NlKSB7XG4gICAgICAgIHJlbW92ZWRQb3Nlcy5wdXNoKHBvc2UpO1xuICAgICAgICBpZiAoZHVwbGljYXRlZFBvc2UuZGVidWcuZHVwbGljYXRlZEl0ZW1zKSB7XG4gICAgICAgICAgZHVwbGljYXRlZFBvc2UuZGVidWcuZHVwbGljYXRlZEl0ZW1zLnB1c2goe1xuICAgICAgICAgICAgaWQ6IHBvc2UuaWQsXG4gICAgICAgICAgICB0aW1lTWlsaXNlY29uZHM6IHBvc2UudGltZU1pbGlzZWNvbmRzLFxuICAgICAgICAgICAgZHVyYXRpb25NaWxpc2Vjb25kczogcG9zZS5kdXJhdGlvbk1pbGlzZWNvbmRzLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuXG4gICAgICBuZXdQb3Nlcy5wdXNoKHBvc2UpO1xuICAgIH1cblxuICAgIGNvbnNvbGUuaW5mbyhcbiAgICAgIGBbUG9zZVNldF0gcmVtb3ZlRHVwbGljYXRlZFBvc2VzIC0gUmVkdWNlZCAke3RoaXMucG9zZXMubGVuZ3RofSBwb3NlcyAtPiAke25ld1Bvc2VzLmxlbmd0aH0gcG9zZXNgLFxuICAgICAge1xuICAgICAgICByZW1vdmVkOiByZW1vdmVkUG9zZXMsXG4gICAgICAgIGtlZXBlZDogbmV3UG9zZXMsXG4gICAgICB9XG4gICAgKTtcbiAgICB0aGlzLnBvc2VzID0gbmV3UG9zZXM7XG4gIH1cblxuICBzdGF0aWMgZ2V0SWRCeVRpbWVNaWxpc2Vjb25kcyh0aW1lTWlsaXNlY29uZHM6IG51bWJlcikge1xuICAgIHJldHVybiBNYXRoLmZsb29yKHRpbWVNaWxpc2Vjb25kcyAvIDEwMCkgKiAxMDA7XG4gIH1cblxuICBwcml2YXRlIGdldEZpbGVFeHRlbnNpb25CeU1pbWUoSU1BR0VfTUlNRTogc3RyaW5nKSB7XG4gICAgc3dpdGNoIChJTUFHRV9NSU1FKSB7XG4gICAgICBjYXNlICdpbWFnZS9wbmcnOlxuICAgICAgICByZXR1cm4gJ3BuZyc7XG4gICAgICBjYXNlICdpbWFnZS9qcGVnJzpcbiAgICAgICAgcmV0dXJuICdqcGcnO1xuICAgICAgY2FzZSAnaW1hZ2Uvd2VicCc6XG4gICAgICAgIHJldHVybiAnd2VicCc7XG4gICAgICBkZWZhdWx0OlxuICAgICAgICByZXR1cm4gJ3BuZyc7XG4gICAgfVxuICB9XG59XG4iXX0=