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
                id: item.id,
                timeMiliseconds: item.timeMiliseconds,
                durationMiliseconds: item.durationMiliseconds,
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1zZXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi9wcm9qZWN0cy9uZ3gtbXAtcG9zZS1leHRyYWN0b3Ivc3JjL2xpYi9jbGFzc2VzL3Bvc2Utc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxjQUFjLEVBQVcsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssS0FBSyxNQUFNLE9BQU8sQ0FBQztBQU0vQixhQUFhO0FBQ2IsT0FBTyxjQUFjLE1BQU0sZ0JBQWdCLENBQUM7QUFDNUMsYUFBYTtBQUNiLE9BQU8sS0FBSyxjQUFjLE1BQU0sZ0JBQWdCLENBQUM7QUFHakQsT0FBTyxFQUFFLFlBQVksRUFBRSxNQUFNLDJCQUEyQixDQUFDO0FBR3pELE1BQU0sT0FBTyxPQUFPO0lBOEVsQjtRQXBFTyxVQUFLLEdBQWtCLEVBQUUsQ0FBQztRQUMxQixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQThDckMsaUJBQWlCO1FBQ1QscUJBQWdCLEdBQWtCLEVBQUUsQ0FBQztRQUU3Qyx1QkFBdUI7UUFDTixrREFBNkMsR0FBRyxJQUFJLENBQUM7UUFFdEUsYUFBYTtRQUNJLGdCQUFXLEdBQVcsSUFBSSxDQUFDO1FBQzNCLGVBQVUsR0FDekIsWUFBWSxDQUFDO1FBQ0Usa0JBQWEsR0FBRyxHQUFHLENBQUM7UUFFckMsVUFBVTtRQUNPLGdDQUEyQixHQUFHLFNBQVMsQ0FBQztRQUN4Qyx5Q0FBb0MsR0FBRyxFQUFFLENBQUM7UUFFM0QsV0FBVztRQUNNLHVDQUFrQyxHQUFHLFNBQVMsQ0FBQztRQUMvQyx1Q0FBa0MsR0FBRyxXQUFXLENBQUM7UUFDakQsNENBQXVDLEdBQUcsR0FBRyxDQUFDO1FBRzdELElBQUksQ0FBQyxhQUFhLEdBQUc7WUFDbkIsSUFBSSxFQUFFLEVBQUU7WUFDUixLQUFLLEVBQUUsQ0FBQztZQUNSLE1BQU0sRUFBRSxDQUFDO1lBQ1QsUUFBUSxFQUFFLENBQUM7WUFDWCxx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L0IsdUNBQXVDO0lBQ3ZDLFVBQVU7SUFDVixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0NBQ3pDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyBQT1NFX0xBTkRNQVJLUywgUmVzdWx0cyB9IGZyb20gJ0BtZWRpYXBpcGUvaG9saXN0aWMnO1xuaW1wb3J0ICogYXMgSlNaaXAgZnJvbSAnanN6aXAnO1xuaW1wb3J0IHsgUG9zZVNldEl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWl0ZW0nO1xuaW1wb3J0IHsgUG9zZVNldEpzb24gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWpzb24nO1xuaW1wb3J0IHsgUG9zZVNldEpzb25JdGVtIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXNldC1qc29uLWl0ZW0nO1xuaW1wb3J0IHsgQm9keVZlY3RvciB9IGZyb20gJy4uL2ludGVyZmFjZXMvYm9keS12ZWN0b3InO1xuXG4vLyBAdHMtaWdub3JlXG5pbXBvcnQgY29zU2ltaWxhcml0eUEgZnJvbSAnY29zLXNpbWlsYXJpdHknO1xuLy8gQHRzLWlnbm9yZVxuaW1wb3J0ICogYXMgY29zU2ltaWxhcml0eUIgZnJvbSAnY29zLXNpbWlsYXJpdHknO1xuXG5pbXBvcnQgeyBTaW1pbGFyUG9zZUl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3NpbWlsYXItcG9zZS1pdGVtJztcbmltcG9ydCB7IEltYWdlVHJpbW1lciB9IGZyb20gJy4vaW50ZXJuYWxzL2ltYWdlLXRyaW1tZXInO1xuaW1wb3J0IHsgSGFuZFZlY3RvciB9IGZyb20gJy4uL2ludGVyZmFjZXMvaGFuZC12ZWN0b3InO1xuXG5leHBvcnQgY2xhc3MgUG9zZVNldCB7XG4gIHB1YmxpYyBnZW5lcmF0b3I/OiBzdHJpbmc7XG4gIHB1YmxpYyB2ZXJzaW9uPzogbnVtYmVyO1xuICBwcml2YXRlIHZpZGVvTWV0YWRhdGEhOiB7XG4gICAgbmFtZTogc3RyaW5nO1xuICAgIHdpZHRoOiBudW1iZXI7XG4gICAgaGVpZ2h0OiBudW1iZXI7XG4gICAgZHVyYXRpb246IG51bWJlcjtcbiAgICBmaXJzdFBvc2VEZXRlY3RlZFRpbWU6IG51bWJlcjtcbiAgfTtcbiAgcHVibGljIHBvc2VzOiBQb3NlU2V0SXRlbVtdID0gW107XG4gIHB1YmxpYyBpc0ZpbmFsaXplZD86IGJvb2xlYW4gPSBmYWxzZTtcblxuICAvLyBCb2R5VmVjdG9yIOOBruOCreODvOWQjVxuICBwdWJsaWMgc3RhdGljIHJlYWRvbmx5IEJPRFlfVkVDVE9SX01BUFBJTkdTID0gW1xuICAgIC8vIOWPs+iFlVxuICAgICdyaWdodFdyaXN0VG9SaWdodEVsYm93JyxcbiAgICAncmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcicsXG4gICAgLy8g5bem6IWVXG4gICAgJ2xlZnRXcmlzdFRvTGVmdEVsYm93JyxcbiAgICAnbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXInLFxuICBdO1xuXG4gIC8vIEhhbmRWZWN0b3Ig44Gu44Kt44O85ZCNXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgSEFORF9WRUNUT1JfTUFQUElOR1MgPSBbXG4gICAgLy8g5Y+z5omLIC0g6Kaq5oyHXG4gICAgJ3JpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOS6uuW3ruOBl+aMh1xuICAgICdyaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDkuK3mjIdcbiAgICAncmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDolqzmjIdcbiAgICAncmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICdyaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAnbGVmdFRodW1iVGlwVG9GaXJzdEpvaW50JyxcbiAgICAnbGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOS6uuW3ruOBl+aMh1xuICAgICdsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g5Lit5oyHXG4gICAgJ2xlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOiWrOaMh1xuICAgICdsZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOWwj+aMh1xuICAgICdsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gIF07XG5cbiAgLy8g44Od44O844K644KS6L+95Yqg44GZ44KL44Gf44KB44Gu44Kt44Ol44O8XG4gIHByaXZhdGUgc2ltaWxhclBvc2VRdWV1ZTogUG9zZVNldEl0ZW1bXSA9IFtdO1xuXG4gIC8vIOmhnuS8vOODneODvOOCuuOBrumZpOWOuyAtIOWQhOODneODvOOCuuOBruWJjeW+jOOBi+OCiVxuICBwcml2YXRlIHJlYWRvbmx5IElTX0VOQUJMRURfUkVNT1ZFX0RVUExJQ0FURURfUE9TRVNfRk9SX0FST1VORCA9IHRydWU7XG5cbiAgLy8g55S75YOP5pu444GN5Ye644GX5pmC44Gu6Kit5a6aXG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfV0lEVEg6IG51bWJlciA9IDEwODA7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfTUlNRTogJ2ltYWdlL2pwZWcnIHwgJ2ltYWdlL3BuZycgfCAnaW1hZ2Uvd2VicCcgPVxuICAgICdpbWFnZS93ZWJwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9RVUFMSVRZID0gMC44O1xuXG4gIC8vIOeUu+WDj+OBruS9meeZvemZpOWOu1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX01BUkdJTl9UUklNTUlOR19DT0xPUiA9ICcjMDAwMDAwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9NQVJHSU5fVFJJTU1JTkdfRElGRl9USFJFU0hPTEQgPSA1MDtcblxuICAvLyDnlLvlg4/jga7og4zmma/oibLnva7mj5tcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfU1JDX0NPTE9SID0gJyMwMTZBRkQnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9EU1RfQ09MT1IgPSAnI0ZGRkZGRjAwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRElGRl9USFJFU0hPTEQgPSAxMzA7XG5cbiAgY29uc3RydWN0b3IoKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhID0ge1xuICAgICAgbmFtZTogJycsXG4gICAgICB3aWR0aDogMCxcbiAgICAgIGhlaWdodDogMCxcbiAgICAgIGR1cmF0aW9uOiAwLFxuICAgICAgZmlyc3RQb3NlRGV0ZWN0ZWRUaW1lOiAwLFxuICAgIH07XG4gIH1cblxuICBnZXRWaWRlb05hbWUoKSB7XG4gICAgcmV0dXJuIHRoaXMudmlkZW9NZXRhZGF0YS5uYW1lO1xuICB9XG5cbiAgc2V0VmlkZW9OYW1lKHZpZGVvTmFtZTogc3RyaW5nKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLm5hbWUgPSB2aWRlb05hbWU7XG4gIH1cblxuICBzZXRWaWRlb01ldGFEYXRhKHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLCBkdXJhdGlvbjogbnVtYmVyKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLndpZHRoID0gd2lkdGg7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLmhlaWdodCA9IGhlaWdodDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24gPSBkdXJhdGlvbjtcbiAgfVxuXG4gIC8qKlxuICAgKiDjg53jg7zjgrrmlbDjga7lj5blvpdcbiAgICogQHJldHVybnNcbiAgICovXG4gIGdldE51bWJlck9mUG9zZXMoKTogbnVtYmVyIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gLTE7XG4gICAgcmV0dXJuIHRoaXMucG9zZXMubGVuZ3RoO1xuICB9XG5cbiAgLyoqXG4gICAqIOWFqOODneODvOOCuuOBruWPluW+l1xuICAgKiBAcmV0dXJucyDlhajjgabjga7jg53jg7zjgrpcbiAgICovXG4gIGdldFBvc2VzKCk6IFBvc2VTZXRJdGVtW10ge1xuICAgIGlmICh0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpIHJldHVybiBbXTtcbiAgICByZXR1cm4gdGhpcy5wb3NlcztcbiAgfVxuXG4gIC8qKlxuICAgKiDmjIflrprjgZXjgozjgZ9JRCAoUG9zZVNldEl0ZW1JZCkg44Gr44KI44KL44Od44O844K644Gu5Y+W5b6XXG4gICAqIEBwYXJhbSBwb3NlU2V0SXRlbUlkXG4gICAqIEByZXR1cm5zIOODneODvOOCulxuICAgKi9cbiAgZ2V0UG9zZUJ5SWQocG9zZVNldEl0ZW1JZDogbnVtYmVyKTogUG9zZVNldEl0ZW0gfCB1bmRlZmluZWQge1xuICAgIGlmICh0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpIHJldHVybiB1bmRlZmluZWQ7XG4gICAgcmV0dXJuIHRoaXMucG9zZXMuZmluZCgocG9zZSkgPT4gcG9zZS5pZCA9PT0gcG9zZVNldEl0ZW1JZCk7XG4gIH1cblxuICAvKipcbiAgICog5oyH5a6a44GV44KM44Gf5pmC6ZaT44Gr44KI44KL44Od44O844K644Gu5Y+W5b6XXG4gICAqIEBwYXJhbSB0aW1lTWlsaXNlY29uZHMg44Od44O844K644Gu5pmC6ZaTICjjg5/jg6rnp5IpXG4gICAqIEByZXR1cm5zIOODneODvOOCulxuICAgKi9cbiAgZ2V0UG9zZUJ5VGltZSh0aW1lTWlsaXNlY29uZHM6IG51bWJlcik6IFBvc2VTZXRJdGVtIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgIHJldHVybiB0aGlzLnBvc2VzLmZpbmQoKHBvc2UpID0+IHBvc2UudGltZU1pbGlzZWNvbmRzID09PSB0aW1lTWlsaXNlY29uZHMpO1xuICB9XG5cbiAgLyoqXG4gICAqIOODneODvOOCuuOBrui/veWKoFxuICAgKi9cbiAgcHVzaFBvc2UoXG4gICAgdmlkZW9UaW1lTWlsaXNlY29uZHM6IG51bWJlcixcbiAgICBmcmFtZUltYWdlRGF0YVVybDogc3RyaW5nIHwgdW5kZWZpbmVkLFxuICAgIHBvc2VJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICBmYWNlRnJhbWVJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICByZXN1bHRzOiBSZXN1bHRzXG4gICk6IFBvc2VTZXRJdGVtIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAodGhpcy5wb3Nlcy5sZW5ndGggPT09IDApIHtcbiAgICAgIHRoaXMudmlkZW9NZXRhZGF0YS5maXJzdFBvc2VEZXRlY3RlZFRpbWUgPSB2aWRlb1RpbWVNaWxpc2Vjb25kcztcbiAgICB9XG5cbiAgICBpZiAocmVzdWx0cy5wb3NlTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHtcbiAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgcG9zZUxhbmRtYXJrc2AsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3QgcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGU6IGFueVtdID0gKHJlc3VsdHMgYXMgYW55KS5lYVxuICAgICAgPyAocmVzdWx0cyBhcyBhbnkpLmVhXG4gICAgICA6IFtdO1xuICAgIGlmIChwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZS5sZW5ndGggPT09IDApIHtcbiAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIHBvc2Ugd2l0aCB0aGUgd29ybGQgY29vcmRpbmF0ZWAsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3QgYm9keVZlY3RvciA9IFBvc2VTZXQuZ2V0Qm9keVZlY3Rvcihwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZSk7XG4gICAgaWYgKCFib2R5VmVjdG9yKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIGJvZHkgdmVjdG9yYCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGVcbiAgICAgICk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgaWYgKFxuICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkICYmXG4gICAgICByZXN1bHRzLnJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkXG4gICAgKSB7XG4gICAgICBjb25zb2xlLmRlYnVnKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBoYW5kIGxhbmRtYXJrc2AsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgfSBlbHNlIGlmIChyZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHtcbiAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIGxlZnQgaGFuZCBsYW5kbWFya3NgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH0gZWxzZSBpZiAocmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCkge1xuICAgICAgY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgcmlnaHQgaGFuZCBsYW5kbWFya3NgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH1cblxuICAgIGNvbnN0IGhhbmRWZWN0b3IgPSBQb3NlU2V0LmdldEhhbmRWZWN0b3IoXG4gICAgICByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzLFxuICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3NcbiAgICApO1xuICAgIGlmICghaGFuZFZlY3Rvcikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBoYW5kIHZlY3RvcmAsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgfVxuXG4gICAgY29uc3QgcG9zZTogUG9zZVNldEl0ZW0gPSB7XG4gICAgICBpZDogUG9zZVNldC5nZXRJZEJ5VGltZU1pbGlzZWNvbmRzKHZpZGVvVGltZU1pbGlzZWNvbmRzKSxcbiAgICAgIHRpbWVNaWxpc2Vjb25kczogdmlkZW9UaW1lTWlsaXNlY29uZHMsXG4gICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiAtMSxcbiAgICAgIHBvc2U6IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlLm1hcCgod29ybGRDb29yZGluYXRlTGFuZG1hcmspID0+IHtcbiAgICAgICAgcmV0dXJuIFtcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay54LFxuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLnksXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsueixcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay52aXNpYmlsaXR5LFxuICAgICAgICBdO1xuICAgICAgfSksXG4gICAgICBsZWZ0SGFuZDogcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcz8ubWFwKChub3JtYWxpemVkTGFuZG1hcmspID0+IHtcbiAgICAgICAgcmV0dXJuIFtcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueCxcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueSxcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueixcbiAgICAgICAgXTtcbiAgICAgIH0pLFxuICAgICAgcmlnaHRIYW5kOiByZXN1bHRzLnJpZ2h0SGFuZExhbmRtYXJrcz8ubWFwKChub3JtYWxpemVkTGFuZG1hcmspID0+IHtcbiAgICAgICAgcmV0dXJuIFtcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueCxcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueSxcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueixcbiAgICAgICAgXTtcbiAgICAgIH0pLFxuICAgICAgYm9keVZlY3RvcjogYm9keVZlY3RvcixcbiAgICAgIGhhbmRWZWN0b3I6IGhhbmRWZWN0b3IsXG4gICAgICBmcmFtZUltYWdlRGF0YVVybDogZnJhbWVJbWFnZURhdGFVcmwsXG4gICAgICBwb3NlSW1hZ2VEYXRhVXJsOiBwb3NlSW1hZ2VEYXRhVXJsLFxuICAgICAgZmFjZUZyYW1lSW1hZ2VEYXRhVXJsOiBmYWNlRnJhbWVJbWFnZURhdGFVcmwsXG4gICAgICBleHRlbmRlZERhdGE6IHt9LFxuICAgICAgZGVidWc6IHtcbiAgICAgICAgZHVwbGljYXRlZEl0ZW1zOiBbXSxcbiAgICAgIH0sXG4gICAgICBtZXJnZWRUaW1lTWlsaXNlY29uZHM6IC0xLFxuICAgICAgbWVyZ2VkRHVyYXRpb25NaWxpc2Vjb25kczogLTEsXG4gICAgfTtcblxuICAgIGxldCBsYXN0UG9zZTtcbiAgICBpZiAodGhpcy5wb3Nlcy5sZW5ndGggPT09IDAgJiYgMSA8PSB0aGlzLnNpbWlsYXJQb3NlUXVldWUubGVuZ3RoKSB7XG4gICAgICAvLyDpoZ7kvLzjg53jg7zjgrrjgq3jg6Xjg7zjgYvjgonmnIDlvozjga7jg53jg7zjgrrjgpLlj5blvpdcbiAgICAgIGxhc3RQb3NlID0gdGhpcy5zaW1pbGFyUG9zZVF1ZXVlW3RoaXMuc2ltaWxhclBvc2VRdWV1ZS5sZW5ndGggLSAxXTtcbiAgICB9IGVsc2UgaWYgKDEgPD0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIC8vIOODneODvOOCuumFjeWIl+OBi+OCieacgOW+jOOBruODneODvOOCuuOCkuWPluW+l1xuICAgICAgbGFzdFBvc2UgPSB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV07XG4gICAgfVxuXG4gICAgaWYgKGxhc3RQb3NlKSB7XG4gICAgICAvLyDmnIDlvozjga7jg53jg7zjgrrjgYzjgYLjgozjgbDjgIHpoZ7kvLzjg53jg7zjgrrjgYvjganjgYbjgYvjgpLmr5TovINcbiAgICAgIGNvbnN0IGlzU2ltaWxhckJvZHlQb3NlID0gUG9zZVNldC5pc1NpbWlsYXJCb2R5UG9zZShcbiAgICAgICAgcG9zZS5ib2R5VmVjdG9yLFxuICAgICAgICBsYXN0UG9zZS5ib2R5VmVjdG9yXG4gICAgICApO1xuXG4gICAgICBsZXQgaXNTaW1pbGFySGFuZFBvc2UgPSB0cnVlO1xuICAgICAgaWYgKGxhc3RQb3NlLmhhbmRWZWN0b3IgJiYgcG9zZS5oYW5kVmVjdG9yKSB7XG4gICAgICAgIGlzU2ltaWxhckhhbmRQb3NlID0gUG9zZVNldC5pc1NpbWlsYXJIYW5kUG9zZShcbiAgICAgICAgICBwb3NlLmhhbmRWZWN0b3IsXG4gICAgICAgICAgbGFzdFBvc2UuaGFuZFZlY3RvclxuICAgICAgICApO1xuICAgICAgfSBlbHNlIGlmICghbGFzdFBvc2UuaGFuZFZlY3RvciAmJiBwb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgaXNTaW1pbGFySGFuZFBvc2UgPSBmYWxzZTtcbiAgICAgIH1cblxuICAgICAgaWYgKCFpc1NpbWlsYXJCb2R5UG9zZSB8fCAhaXNTaW1pbGFySGFuZFBvc2UpIHtcbiAgICAgICAgLy8g6Lqr5L2T44O75omL44Gu44GE44Ga44KM44GL44GM5YmN44Gu44Od44O844K644Go6aGe5Ly844GX44Gm44GE44Gq44GE44Gq44KJ44Gw44CB6aGe5Ly844Od44O844K644Kt44Ol44O844KS5Yem55CG44GX44Gm44CB44Od44O844K66YWN5YiX44G46L+95YqgXG4gICAgICAgIHRoaXMucHVzaFBvc2VGcm9tU2ltaWxhclBvc2VRdWV1ZShwb3NlLnRpbWVNaWxpc2Vjb25kcyk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8g6aGe5Ly844Od44O844K644Kt44Ol44O844G46L+95YqgXG4gICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlLnB1c2gocG9zZSk7XG5cbiAgICByZXR1cm4gcG9zZTtcbiAgfVxuXG4gIC8qKlxuICAgKiDjg53jg7zjgrrjga7phY3liJfjgYvjgonjg53jg7zjgrrjgYzmsbrjgb7jgaPjgabjgYTjgovnnqzplpPjgpLlj5blvpdcbiAgICogQHBhcmFtIHBvc2VzIOODneODvOOCuuOBrumFjeWIl1xuICAgKiBAcmV0dXJucyDjg53jg7zjgrrjgYzmsbrjgb7jgaPjgabjgYTjgovnnqzplpNcbiAgICovXG4gIHN0YXRpYyBnZXRTdWl0YWJsZVBvc2VCeVBvc2VzKHBvc2VzOiBQb3NlU2V0SXRlbVtdKTogUG9zZVNldEl0ZW0ge1xuICAgIGlmIChwb3Nlcy5sZW5ndGggPT09IDApIHJldHVybiBudWxsO1xuICAgIGlmIChwb3Nlcy5sZW5ndGggPT09IDEpIHtcbiAgICAgIHJldHVybiBwb3Nlc1sxXTtcbiAgICB9XG5cbiAgICAvLyDlkITmqJnmnKzjg53jg7zjgrrjgZTjgajjga7poZ7kvLzluqbjgpLliJ3mnJ/ljJZcbiAgICBjb25zdCBzaW1pbGFyaXRpZXNPZlBvc2VzOiB7XG4gICAgICBba2V5OiBudW1iZXJdOiB7XG4gICAgICAgIGhhbmRTaW1pbGFyaXR5OiBudW1iZXI7XG4gICAgICAgIGJvZHlTaW1pbGFyaXR5OiBudW1iZXI7XG4gICAgICB9W107XG4gICAgfSA9IHt9O1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcG9zZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgIHNpbWlsYXJpdGllc09mUG9zZXNbcG9zZXNbaV0udGltZU1pbGlzZWNvbmRzXSA9IHBvc2VzLm1hcChcbiAgICAgICAgKHBvc2U6IFBvc2VTZXRJdGVtKSA9PiB7XG4gICAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICAgIGhhbmRTaW1pbGFyaXR5OiAwLFxuICAgICAgICAgICAgYm9keVNpbWlsYXJpdHk6IDAsXG4gICAgICAgICAgfTtcbiAgICAgICAgfVxuICAgICAgKTtcbiAgICB9XG5cbiAgICAvLyDlkITmqJnmnKzjg53jg7zjgrrjgZTjgajjga7poZ7kvLzluqbjgpLoqIjnrpdcbiAgICBmb3IgKGxldCBzYW1wbGVQb3NlIG9mIHBvc2VzKSB7XG4gICAgICBsZXQgaGFuZFNpbWlsYXJpdHk6IG51bWJlcjtcblxuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBwb3Nlcy5sZW5ndGg7IGkrKykge1xuICAgICAgICBjb25zdCBwb3NlID0gcG9zZXNbaV07XG4gICAgICAgIGlmIChwb3NlLmhhbmRWZWN0b3IgJiYgc2FtcGxlUG9zZS5oYW5kVmVjdG9yKSB7XG4gICAgICAgICAgaGFuZFNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEhhbmRTaW1pbGFyaXR5KFxuICAgICAgICAgICAgcG9zZS5oYW5kVmVjdG9yLFxuICAgICAgICAgICAgc2FtcGxlUG9zZS5oYW5kVmVjdG9yXG4gICAgICAgICAgKTtcbiAgICAgICAgfVxuXG4gICAgICAgIGxldCBib2R5U2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0Qm9keVBvc2VTaW1pbGFyaXR5KFxuICAgICAgICAgIHBvc2UuYm9keVZlY3RvcixcbiAgICAgICAgICBzYW1wbGVQb3NlLmJvZHlWZWN0b3JcbiAgICAgICAgKTtcblxuICAgICAgICBzaW1pbGFyaXRpZXNPZlBvc2VzW3NhbXBsZVBvc2UudGltZU1pbGlzZWNvbmRzXVtpXSA9IHtcbiAgICAgICAgICBoYW5kU2ltaWxhcml0eTogaGFuZFNpbWlsYXJpdHkgPz8gMCxcbiAgICAgICAgICBib2R5U2ltaWxhcml0eSxcbiAgICAgICAgfTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyDpoZ7kvLzluqbjga7pq5jjgYTjg5Xjg6zjg7zjg6DjgYzlpJrjgYvjgaPjgZ/jg53jg7zjgrrjgpLpgbjmip5cbiAgICBjb25zdCBzaW1pbGFyaXRpZXNPZlNhbXBsZVBvc2VzID0gcG9zZXMubWFwKChwb3NlOiBQb3NlU2V0SXRlbSkgPT4ge1xuICAgICAgcmV0dXJuIHNpbWlsYXJpdGllc09mUG9zZXNbcG9zZS50aW1lTWlsaXNlY29uZHNdLnJlZHVjZShcbiAgICAgICAgKFxuICAgICAgICAgIHByZXY6IG51bWJlcixcbiAgICAgICAgICBjdXJyZW50OiB7IGhhbmRTaW1pbGFyaXR5OiBudW1iZXI7IGJvZHlTaW1pbGFyaXR5OiBudW1iZXIgfVxuICAgICAgICApID0+IHtcbiAgICAgICAgICByZXR1cm4gcHJldiArIGN1cnJlbnQuaGFuZFNpbWlsYXJpdHkgKyBjdXJyZW50LmJvZHlTaW1pbGFyaXR5O1xuICAgICAgICB9LFxuICAgICAgICAwXG4gICAgICApO1xuICAgIH0pO1xuICAgIGNvbnN0IG1heFNpbWlsYXJpdHkgPSBNYXRoLm1heCguLi5zaW1pbGFyaXRpZXNPZlNhbXBsZVBvc2VzKTtcbiAgICBjb25zdCBtYXhTaW1pbGFyaXR5SW5kZXggPSBzaW1pbGFyaXRpZXNPZlNhbXBsZVBvc2VzLmluZGV4T2YobWF4U2ltaWxhcml0eSk7XG4gICAgY29uc3Qgc2VsZWN0ZWRQb3NlID0gcG9zZXNbbWF4U2ltaWxhcml0eUluZGV4XTtcbiAgICBpZiAoIXNlbGVjdGVkUG9zZSkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIGdldFN1aXRhYmxlUG9zZUJ5UG9zZXNgLFxuICAgICAgICBzaW1pbGFyaXRpZXNPZlNhbXBsZVBvc2VzLFxuICAgICAgICBtYXhTaW1pbGFyaXR5LFxuICAgICAgICBtYXhTaW1pbGFyaXR5SW5kZXhcbiAgICAgICk7XG4gICAgfVxuXG4gICAgY29uc29sZS5kZWJ1ZyhgW1Bvc2VTZXRdIGdldFN1aXRhYmxlUG9zZUJ5UG9zZXNgLCB7XG4gICAgICBzZWxlY3RlZDogc2VsZWN0ZWRQb3NlLFxuICAgICAgdW5zZWxlY3RlZDogcG9zZXMuZmlsdGVyKChwb3NlOiBQb3NlU2V0SXRlbSkgPT4ge1xuICAgICAgICByZXR1cm4gcG9zZS50aW1lTWlsaXNlY29uZHMgIT09IHNlbGVjdGVkUG9zZS50aW1lTWlsaXNlY29uZHM7XG4gICAgICB9KSxcbiAgICB9KTtcbiAgICByZXR1cm4gc2VsZWN0ZWRQb3NlO1xuICB9XG5cbiAgLyoqXG4gICAqIOacgOe1guWHpueQhlxuICAgKiAo6YeN6KSH44GX44Gf44Od44O844K644Gu6Zmk5Y6744CB55S75YOP44Gu44Oe44O844K444Oz6Zmk5Y6744Gq44GpKVxuICAgKi9cbiAgYXN5bmMgZmluYWxpemUoaXNSZW1vdmVEdXBsaWNhdGU6IGJvb2xlYW4gPSB0cnVlKSB7XG4gICAgaWYgKHRoaXMuc2ltaWxhclBvc2VRdWV1ZS5sZW5ndGggPiAwKSB7XG4gICAgICAvLyDpoZ7kvLzjg53jg7zjgrrjgq3jg6Xjg7zjgavjg53jg7zjgrrjgYzmrovjgaPjgabjgYTjgovloLTlkIjjgIHmnIDpganjgarjg53jg7zjgrrjgpLpgbjmip7jgZfjgabjg53jg7zjgrrphY3liJfjgbjov73liqBcbiAgICAgIHRoaXMucHVzaFBvc2VGcm9tU2ltaWxhclBvc2VRdWV1ZSh0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24pO1xuICAgIH1cblxuICAgIGlmICgwID09IHRoaXMucG9zZXMubGVuZ3RoKSB7XG4gICAgICAvLyDjg53jg7zjgrrjgYzkuIDjgaTjgoLjgarjgYTloLTlkIjjgIHlh6bnkIbjgpLntYLkuoZcbiAgICAgIHRoaXMuaXNGaW5hbGl6ZWQgPSB0cnVlO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIC8vIOODneODvOOCuuOBruaMgee2muaZgumWk+OCkuioreWumlxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5wb3Nlcy5sZW5ndGggLSAxOyBpKyspIHtcbiAgICAgIGlmICh0aGlzLnBvc2VzW2ldLmR1cmF0aW9uTWlsaXNlY29uZHMgIT09IC0xKSBjb250aW51ZTtcbiAgICAgIHRoaXMucG9zZXNbaV0uZHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgIHRoaXMucG9zZXNbaSArIDFdLnRpbWVNaWxpc2Vjb25kcyAtIHRoaXMucG9zZXNbaV0udGltZU1pbGlzZWNvbmRzO1xuICAgIH1cbiAgICB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV0uZHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24gLVxuICAgICAgdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdLnRpbWVNaWxpc2Vjb25kcztcblxuICAgIC8vIOWFqOS9k+OBi+OCiemHjeikh+ODneODvOOCuuOCkumZpOWOu1xuICAgIGlmIChpc1JlbW92ZUR1cGxpY2F0ZSkge1xuICAgICAgdGhpcy5yZW1vdmVEdXBsaWNhdGVkUG9zZXMoKTtcbiAgICB9XG5cbiAgICAvLyDmnIDliJ3jga7jg53jg7zjgrrjgpLpmaTljrtcbiAgICB0aGlzLnBvc2VzLnNoaWZ0KCk7XG5cbiAgICAvLyDnlLvlg4/jga7jg57jg7zjgrjjg7PjgpLlj5blvpdcbiAgICBjb25zb2xlLmRlYnVnKGBbUG9zZVNldF0gZmluYWxpemUgLSBEZXRlY3RpbmcgaW1hZ2UgbWFyZ2lucy4uLmApO1xuICAgIGxldCBpbWFnZVRyaW1taW5nOlxuICAgICAgfCB7XG4gICAgICAgICAgbWFyZ2luVG9wOiBudW1iZXI7XG4gICAgICAgICAgbWFyZ2luQm90dG9tOiBudW1iZXI7XG4gICAgICAgICAgaGVpZ2h0TmV3OiBudW1iZXI7XG4gICAgICAgICAgaGVpZ2h0T2xkOiBudW1iZXI7XG4gICAgICAgICAgd2lkdGg6IG51bWJlcjtcbiAgICAgICAgfVxuICAgICAgfCB1bmRlZmluZWQgPSB1bmRlZmluZWQ7XG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGxldCBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgY29uc3QgbWFyZ2luQ29sb3IgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0TWFyZ2luQ29sb3IoKTtcbiAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBEZXRlY3RlZCBtYXJnaW4gY29sb3IuLi5gLFxuICAgICAgICBwb3NlLnRpbWVNaWxpc2Vjb25kcyxcbiAgICAgICAgbWFyZ2luQ29sb3JcbiAgICAgICk7XG4gICAgICBpZiAobWFyZ2luQ29sb3IgPT09IG51bGwpIGNvbnRpbnVlO1xuICAgICAgaWYgKG1hcmdpbkNvbG9yICE9PSB0aGlzLklNQUdFX01BUkdJTl9UUklNTUlOR19DT0xPUikge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHRyaW1tZWQgPSBhd2FpdCBpbWFnZVRyaW1tZXIudHJpbU1hcmdpbihcbiAgICAgICAgbWFyZ2luQ29sb3IsXG4gICAgICAgIHRoaXMuSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0RJRkZfVEhSRVNIT0xEXG4gICAgICApO1xuICAgICAgaWYgKCF0cmltbWVkKSBjb250aW51ZTtcbiAgICAgIGltYWdlVHJpbW1pbmcgPSB0cmltbWVkO1xuICAgICAgY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVybWluZWQgaW1hZ2UgdHJpbW1pbmcgcG9zaXRpb25zLi4uYCxcbiAgICAgICAgdHJpbW1lZFxuICAgICAgKTtcbiAgICAgIGJyZWFrO1xuICAgIH1cblxuICAgIC8vIOeUu+WDj+OCkuaVtOW9olxuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsIHx8ICFwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG5cbiAgICAgIGNvbnNvbGUuZGVidWcoXG4gICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBQcm9jZXNzaW5nIGltYWdlLi4uYCxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHNcbiAgICAgICk7XG5cbiAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOODleODrOODvOODoOeUu+WDj1xuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5mcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGlmIChpbWFnZVRyaW1taW5nKSB7XG4gICAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5jcm9wKFxuICAgICAgICAgIDAsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5tYXJnaW5Ub3AsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy53aWR0aCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLmhlaWdodE5ld1xuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVwbGFjZUNvbG9yKFxuICAgICAgICB0aGlzLklNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9TUkNfQ09MT1IsXG4gICAgICAgIHRoaXMuSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RTVF9DT0xPUixcbiAgICAgICAgdGhpcy5JTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRElGRl9USFJFU0hPTERcbiAgICAgICk7XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXNpemVXaXRoRml0KHtcbiAgICAgICAgd2lkdGg6IHRoaXMuSU1BR0VfV0lEVEgsXG4gICAgICB9KTtcblxuICAgICAgbGV0IG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgID8gdGhpcy5JTUFHRV9RVUFMSVRZXG4gICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICk7XG4gICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgZnJhbWUgaW1hZ2VgXG4gICAgICAgICk7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG5cbiAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOODneODvOOCuuODl+ODrOODk+ODpeODvOeUu+WDj1xuICAgICAgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5wb3NlSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgaWYgKGltYWdlVHJpbW1pbmcpIHtcbiAgICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmNyb3AoXG4gICAgICAgICAgMCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLm1hcmdpblRvcCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLndpZHRoLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcuaGVpZ2h0TmV3XG4gICAgICAgICk7XG4gICAgICB9XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXNpemVXaXRoRml0KHtcbiAgICAgICAgd2lkdGg6IHRoaXMuSU1BR0VfV0lEVEgsXG4gICAgICB9KTtcblxuICAgICAgbmV3RGF0YVVybCA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXREYXRhVXJsKFxuICAgICAgICB0aGlzLklNQUdFX01JTUUsXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL2pwZWcnIHx8IHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL3dlYnAnXG4gICAgICAgICAgPyB0aGlzLklNQUdFX1FVQUxJVFlcbiAgICAgICAgICA6IHVuZGVmaW5lZFxuICAgICAgKTtcbiAgICAgIGlmICghbmV3RGF0YVVybCkge1xuICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIENvdWxkIG5vdCBnZXQgdGhlIG5ldyBkYXRhdXJsIGZvciBwb3NlIHByZXZpZXcgaW1hZ2VgXG4gICAgICAgICk7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsID0gbmV3RGF0YVVybDtcblxuICAgICAgaWYgKHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOmhlOODleODrOODvOODoOeUu+WDj1xuICAgICAgICBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgICBuZXdEYXRhVXJsID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldERhdGFVcmwoXG4gICAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICAgIHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL2pwZWcnIHx8IHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL3dlYnAnXG4gICAgICAgICAgICA/IHRoaXMuSU1BR0VfUVVBTElUWVxuICAgICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICAgKTtcbiAgICAgICAgaWYgKCFuZXdEYXRhVXJsKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIENvdWxkIG5vdCBnZXQgdGhlIG5ldyBkYXRhdXJsIGZvciBmYWNlIGZyYW1lIGltYWdlYFxuICAgICAgICAgICk7XG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cbiAgICAgICAgcG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwgPSBuZXdEYXRhVXJsO1xuICAgICAgfVxuICAgIH1cblxuICAgIHRoaXMuaXNGaW5hbGl6ZWQgPSB0cnVlO1xuICB9XG5cbiAgLyoqXG4gICAqIOmhnuS8vOODneODvOOCuuOBruWPluW+l1xuICAgKiBAcGFyYW0gcmVzdWx0cyBNZWRpYVBpcGUgSG9saXN0aWMg44Gr44KI44KL44Od44O844K644Gu5qSc5Ye657WQ5p6cXG4gICAqIEBwYXJhbSB0aHJlc2hvbGQg44GX44GN44GE5YCkXG4gICAqIEBwYXJhbSB0YXJnZXRSYW5nZSDjg53jg7zjgrrjgpLmr5TovIPjgZnjgovnr4Tlm7IgKGFsbDog5YWo44GmLCBib2R5UG9zZTog6Lqr5L2T44Gu44G/LCBoYW5kUG9zZTog5omL5oyH44Gu44G/KVxuICAgKiBAcmV0dXJucyDpoZ7kvLzjg53jg7zjgrrjga7phY3liJdcbiAgICovXG4gIGdldFNpbWlsYXJQb3NlcyhcbiAgICByZXN1bHRzOiBSZXN1bHRzLFxuICAgIHRocmVzaG9sZDogbnVtYmVyID0gMC45LFxuICAgIHRhcmdldFJhbmdlOiAnYWxsJyB8ICdib2R5UG9zZScgfCAnaGFuZFBvc2UnID0gJ2FsbCdcbiAgKTogU2ltaWxhclBvc2VJdGVtW10ge1xuICAgIC8vIOi6q+S9k+OBruODmeOCr+ODiOODq+OCkuWPluW+l1xuICAgIGxldCBib2R5VmVjdG9yOiBCb2R5VmVjdG9yO1xuICAgIHRyeSB7XG4gICAgICBib2R5VmVjdG9yID0gUG9zZVNldC5nZXRCb2R5VmVjdG9yKChyZXN1bHRzIGFzIGFueSkuZWEpO1xuICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgIGNvbnNvbGUuZXJyb3IoYFtQb3NlU2V0XSBnZXRTaW1pbGFyUG9zZXMgLSBFcnJvciBvY2N1cnJlZGAsIGUsIHJlc3VsdHMpO1xuICAgICAgcmV0dXJuIFtdO1xuICAgIH1cbiAgICBpZiAoIWJvZHlWZWN0b3IpIHtcbiAgICAgIHRocm93ICdDb3VsZCBub3QgZ2V0IHRoZSBib2R5IHZlY3Rvcic7XG4gICAgfVxuXG4gICAgLy8g5omL5oyH44Gu44OZ44Kv44OI44Or44KS5Y+W5b6XXG4gICAgbGV0IGhhbmRWZWN0b3I6IEhhbmRWZWN0b3I7XG4gICAgaWYgKHRhcmdldFJhbmdlID09PSAnYWxsJyB8fCB0YXJnZXRSYW5nZSA9PT0gJ2hhbmRQb3NlJykge1xuICAgICAgaGFuZFZlY3RvciA9IFBvc2VTZXQuZ2V0SGFuZFZlY3RvcihcbiAgICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyxcbiAgICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3NcbiAgICAgICk7XG4gICAgICBpZiAodGFyZ2V0UmFuZ2UgPT09ICdoYW5kUG9zZScgJiYgIWhhbmRWZWN0b3IpIHtcbiAgICAgICAgdGhyb3cgJ0NvdWxkIG5vdCBnZXQgdGhlIGhhbmQgdmVjdG9yJztcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyDlkITjg53jg7zjgrrjgajjg5njgq/jg4jjg6vjgpLmr5TovINcbiAgICBjb25zdCBwb3NlcyA9IFtdO1xuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBpZiAoXG4gICAgICAgICh0YXJnZXRSYW5nZSA9PT0gJ2FsbCcgfHwgdGFyZ2V0UmFuZ2UgPT09ICdib2R5UG9zZScpICYmXG4gICAgICAgICFwb3NlLmJvZHlWZWN0b3JcbiAgICAgICkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH0gZWxzZSBpZiAodGFyZ2V0UmFuZ2UgPT09ICdoYW5kUG9zZScgJiYgIXBvc2UuaGFuZFZlY3Rvcikge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cblxuICAgICAgLypjb25zb2xlLmRlYnVnKFxuICAgICAgICAnW1Bvc2VTZXRdIGdldFNpbWlsYXJQb3NlcyAtICcsXG4gICAgICAgIHRoaXMuZ2V0VmlkZW9OYW1lKCksXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzXG4gICAgICApOyovXG5cbiAgICAgIC8vIOi6q+S9k+OBruODneODvOOCuuOBrumhnuS8vOW6puOCkuWPluW+l1xuICAgICAgbGV0IGJvZHlTaW1pbGFyaXR5OiBudW1iZXI7XG4gICAgICBpZiAoYm9keVZlY3RvciAmJiBwb3NlLmJvZHlWZWN0b3IpIHtcbiAgICAgICAgYm9keVNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEJvZHlQb3NlU2ltaWxhcml0eShcbiAgICAgICAgICBwb3NlLmJvZHlWZWN0b3IsXG4gICAgICAgICAgYm9keVZlY3RvclxuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICAvLyDmiYvmjIfjga7jg53jg7zjgrrjga7poZ7kvLzluqbjgpLlj5blvpdcbiAgICAgIGxldCBoYW5kU2ltaWxhcml0eTogbnVtYmVyO1xuICAgICAgaWYgKGhhbmRWZWN0b3IgJiYgcG9zZS5oYW5kVmVjdG9yKSB7XG4gICAgICAgIGhhbmRTaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRIYW5kU2ltaWxhcml0eShwb3NlLmhhbmRWZWN0b3IsIGhhbmRWZWN0b3IpO1xuICAgICAgfVxuXG4gICAgICAvLyDliKTlrppcbiAgICAgIGxldCBzaW1pbGFyaXR5OiBudW1iZXIsXG4gICAgICAgIGlzU2ltaWxhciA9IGZhbHNlO1xuICAgICAgaWYgKHRhcmdldFJhbmdlID09PSAnYWxsJykge1xuICAgICAgICBzaW1pbGFyaXR5ID0gTWF0aC5tYXgoYm9keVNpbWlsYXJpdHkgPz8gMCwgaGFuZFNpbWlsYXJpdHkgPz8gMCk7XG4gICAgICAgIGlmICh0aHJlc2hvbGQgPD0gYm9keVNpbWlsYXJpdHkgfHwgdGhyZXNob2xkIDw9IGhhbmRTaW1pbGFyaXR5KSB7XG4gICAgICAgICAgaXNTaW1pbGFyID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2JvZHlQb3NlJykge1xuICAgICAgICBzaW1pbGFyaXR5ID0gYm9keVNpbWlsYXJpdHk7XG4gICAgICAgIGlmICh0aHJlc2hvbGQgPD0gYm9keVNpbWlsYXJpdHkpIHtcbiAgICAgICAgICBpc1NpbWlsYXIgPSB0cnVlO1xuICAgICAgICB9XG4gICAgICB9IGVsc2UgaWYgKHRhcmdldFJhbmdlID09PSAnaGFuZFBvc2UnKSB7XG4gICAgICAgIHNpbWlsYXJpdHkgPSBoYW5kU2ltaWxhcml0eTtcbiAgICAgICAgaWYgKHRocmVzaG9sZCA8PSBoYW5kU2ltaWxhcml0eSkge1xuICAgICAgICAgIGlzU2ltaWxhciA9IHRydWU7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgaWYgKCFpc1NpbWlsYXIpIGNvbnRpbnVlO1xuXG4gICAgICAvLyDntZDmnpzjgbjov73liqBcbiAgICAgIHBvc2VzLnB1c2goe1xuICAgICAgICAuLi5wb3NlLFxuICAgICAgICBzaW1pbGFyaXR5OiBzaW1pbGFyaXR5LFxuICAgICAgICBib2R5UG9zZVNpbWlsYXJpdHk6IGJvZHlTaW1pbGFyaXR5LFxuICAgICAgICBoYW5kUG9zZVNpbWlsYXJpdHk6IGhhbmRTaW1pbGFyaXR5LFxuICAgICAgfSBhcyBTaW1pbGFyUG9zZUl0ZW0pO1xuICAgIH1cblxuICAgIHJldHVybiBwb3NlcztcbiAgfVxuXG4gIC8qKlxuICAgKiDouqvkvZPjga7lp7/li6LjgpLooajjgZnjg5njgq/jg4jjg6vjga7lj5blvpdcbiAgICogQHBhcmFtIHBvc2VMYW5kbWFya3MgTWVkaWFQaXBlIEhvbGlzdGljIOOBp+WPluW+l+OBp+OBjeOBn+i6q+S9k+OBruODr+ODvOODq+ODieW6p+aomSAocmEg6YWN5YiXKVxuICAgKiBAcmV0dXJucyDjg5njgq/jg4jjg6tcbiAgICovXG4gIHN0YXRpYyBnZXRCb2R5VmVjdG9yKFxuICAgIHBvc2VMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W11cbiAgKTogQm9keVZlY3RvciB8IHVuZGVmaW5lZCB7XG4gICAgcmV0dXJuIHtcbiAgICAgIHJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICByaWdodEVsYm93VG9SaWdodFNob3VsZGVyOiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnosXG4gICAgICBdLFxuICAgICAgbGVmdFdyaXN0VG9MZWZ0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS56LFxuICAgICAgXSxcbiAgICB9O1xuICB9XG5cbiAgLyoqXG4gICAqIOaJi+aMh+OBruWnv+WLouOCkuihqOOBmeODmeOCr+ODiOODq+OBruWPluW+l1xuICAgKiBAcGFyYW0gbGVmdEhhbmRMYW5kbWFya3MgTWVkaWFQaXBlIEhvbGlzdGljIOOBp+WPluW+l+OBp+OBjeOBn+W3puaJi+OBruato+imj+WMluW6p+aomVxuICAgKiBAcGFyYW0gcmlnaHRIYW5kTGFuZG1hcmtzIE1lZGlhUGlwZSBIb2xpc3RpYyDjgaflj5blvpfjgafjgY3jgZ/lj7PmiYvjga7mraPopo/ljJbluqfmqJlcbiAgICogQHJldHVybnMg44OZ44Kv44OI44OrXG4gICAqL1xuICBzdGF0aWMgZ2V0SGFuZFZlY3RvcihcbiAgICBsZWZ0SGFuZExhbmRtYXJrczogeyB4OiBudW1iZXI7IHk6IG51bWJlcjsgejogbnVtYmVyIH1bXSxcbiAgICByaWdodEhhbmRMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W11cbiAgKTogSGFuZFZlY3RvciB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKFxuICAgICAgKHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDApICYmXG4gICAgICAobGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDApXG4gICAgKSB7XG4gICAgICByZXR1cm4gdW5kZWZpbmVkO1xuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICAvLyDlj7PmiYsgLSDopqrmjIdcbiAgICAgIHJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnggLSByaWdodEhhbmRMYW5kbWFya3NbM10ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbM10ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnogLSByaWdodEhhbmRMYW5kbWFya3NbM10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgIHJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnggLSByaWdodEhhbmRMYW5kbWFya3NbN10ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbN10ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnogLSByaWdodEhhbmRMYW5kbWFya3NbN10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnggLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnkgLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnogLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDkuK3mjIdcbiAgICAgIHJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMl0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzEyXS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzExXS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTJdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTFdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTFdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTBdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMF0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzExXS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzEwXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOiWrOaMh1xuICAgICAgcmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNl0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE2XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTZdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTVdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE0XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTVdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTRdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDlsI/mjIdcbiAgICAgIHJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzIwXS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMjBdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTldLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1syMF0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE4XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTldLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMThdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAgIGxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbNF0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzNdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzRdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1szXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s0XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbM10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbM10ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzJdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1syXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1szXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgIGxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbOF0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzddLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzhdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1s3XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s4XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbN10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbN10ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzZdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzddLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1s2XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s3XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbNl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDkuK3mjIdcbiAgICAgIGxlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOiWrOaMh1xuICAgICAgbGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDlsI/mjIdcbiAgICAgIGxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgfTtcbiAgfVxuXG4gIC8qKlxuICAgKiBCb2R5VmVjdG9yIOmWk+OBjOmhnuS8vOOBl+OBpuOBhOOCi+OBi+OBqeOBhuOBi+OBruWIpOWumlxuICAgKiBAcGFyYW0gYm9keVZlY3RvckEg5q+U6LyD5YWI44GuIEJvZHlWZWN0b3JcbiAgICogQHBhcmFtIGJvZHlWZWN0b3JCIOavlOi8g+WFg+OBriBCb2R5VmVjdG9yXG4gICAqIEBwYXJhbSB0aHJlc2hvbGQg44GX44GN44GE5YCkXG4gICAqIEByZXR1cm5zIOmhnuS8vOOBl+OBpuOBhOOCi+OBi+OBqeOBhuOBi1xuICAgKi9cbiAgc3RhdGljIGlzU2ltaWxhckJvZHlQb3NlKFxuICAgIGJvZHlWZWN0b3JBOiBCb2R5VmVjdG9yLFxuICAgIGJvZHlWZWN0b3JCOiBCb2R5VmVjdG9yLFxuICAgIHRocmVzaG9sZCA9IDAuOFxuICApOiBib29sZWFuIHtcbiAgICBsZXQgaXNTaW1pbGFyID0gZmFsc2U7XG4gICAgY29uc3Qgc2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0Qm9keVBvc2VTaW1pbGFyaXR5KGJvZHlWZWN0b3JBLCBib2R5VmVjdG9yQik7XG4gICAgaWYgKHNpbWlsYXJpdHkgPj0gdGhyZXNob2xkKSBpc1NpbWlsYXIgPSB0cnVlO1xuXG4gICAgLy8gY29uc29sZS5kZWJ1ZyhgW1Bvc2VTZXRdIGlzU2ltaWxhclBvc2VgLCBpc1NpbWlsYXIsIHNpbWlsYXJpdHkpO1xuXG4gICAgcmV0dXJuIGlzU2ltaWxhcjtcbiAgfVxuXG4gIC8qKlxuICAgKiDouqvkvZPjg53jg7zjgrrjga7poZ7kvLzluqbjga7lj5blvpdcbiAgICogQHBhcmFtIGJvZHlWZWN0b3JBIOavlOi8g+WFiOOBriBCb2R5VmVjdG9yXG4gICAqIEBwYXJhbSBib2R5VmVjdG9yQiDmr5TovIPlhYPjga4gQm9keVZlY3RvclxuICAgKiBAcmV0dXJucyDpoZ7kvLzluqZcbiAgICovXG4gIHN0YXRpYyBnZXRCb2R5UG9zZVNpbWlsYXJpdHkoXG4gICAgYm9keVZlY3RvckE6IEJvZHlWZWN0b3IsXG4gICAgYm9keVZlY3RvckI6IEJvZHlWZWN0b3JcbiAgKTogbnVtYmVyIHtcbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXMgPSB7XG4gICAgICBsZWZ0V3Jpc3RUb0xlZnRFbGJvdzogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5sZWZ0V3Jpc3RUb0xlZnRFbGJvdyxcbiAgICAgICAgYm9keVZlY3RvckIubGVmdFdyaXN0VG9MZWZ0RWxib3dcbiAgICAgICksXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5sZWZ0RWxib3dUb0xlZnRTaG91bGRlcixcbiAgICAgICAgYm9keVZlY3RvckIubGVmdEVsYm93VG9MZWZ0U2hvdWxkZXJcbiAgICAgICksXG4gICAgICByaWdodFdyaXN0VG9SaWdodEVsYm93OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLnJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3csXG4gICAgICAgIGJvZHlWZWN0b3JCLnJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3dcbiAgICAgICksXG4gICAgICByaWdodEVsYm93VG9SaWdodFNob3VsZGVyOiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXIsXG4gICAgICAgIGJvZHlWZWN0b3JCLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXJcbiAgICAgICksXG4gICAgfTtcblxuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllc1N1bSA9IE9iamVjdC52YWx1ZXMoY29zU2ltaWxhcml0aWVzKS5yZWR1Y2UoXG4gICAgICAoc3VtLCB2YWx1ZSkgPT4gc3VtICsgdmFsdWUsXG4gICAgICAwXG4gICAgKTtcbiAgICByZXR1cm4gY29zU2ltaWxhcml0aWVzU3VtIC8gT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzKS5sZW5ndGg7XG4gIH1cblxuICAvKipcbiAgICogSGFuZFZlY3RvciDplpPjgYzpoZ7kvLzjgZfjgabjgYTjgovjgYvjganjgYbjgYvjga7liKTlrppcbiAgICogQHBhcmFtIGhhbmRWZWN0b3JBIOavlOi8g+WFiOOBriBIYW5kVmVjdG9yXG4gICAqIEBwYXJhbSBoYW5kVmVjdG9yQiDmr5TovIPlhYPjga4gSGFuZFZlY3RvclxuICAgKiBAcGFyYW0gdGhyZXNob2xkIOOBl+OBjeOBhOWApFxuICAgKiBAcmV0dXJucyDpoZ7kvLzjgZfjgabjgYTjgovjgYvjganjgYbjgYtcbiAgICovXG4gIHN0YXRpYyBpc1NpbWlsYXJIYW5kUG9zZShcbiAgICBoYW5kVmVjdG9yQTogSGFuZFZlY3RvcixcbiAgICBoYW5kVmVjdG9yQjogSGFuZFZlY3RvcixcbiAgICB0aHJlc2hvbGQgPSAwLjc1XG4gICk6IGJvb2xlYW4ge1xuICAgIGNvbnN0IHNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEhhbmRTaW1pbGFyaXR5KGhhbmRWZWN0b3JBLCBoYW5kVmVjdG9yQik7XG4gICAgaWYgKHNpbWlsYXJpdHkgPT09IC0xKSB7XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG4gICAgcmV0dXJuIHNpbWlsYXJpdHkgPj0gdGhyZXNob2xkO1xuICB9XG5cbiAgLyoqXG4gICAqIOaJi+OBruODneODvOOCuuOBrumhnuS8vOW6puOBruWPluW+l1xuICAgKiBAcGFyYW0gaGFuZFZlY3RvckEg5q+U6LyD5YWI44GuIEhhbmRWZWN0b3JcbiAgICogQHBhcmFtIGhhbmRWZWN0b3JCIOavlOi8g+WFg+OBriBIYW5kVmVjdG9yXG4gICAqIEByZXR1cm5zIOmhnuS8vOW6plxuICAgKi9cbiAgc3RhdGljIGdldEhhbmRTaW1pbGFyaXR5KFxuICAgIGhhbmRWZWN0b3JBOiBIYW5kVmVjdG9yLFxuICAgIGhhbmRWZWN0b3JCOiBIYW5kVmVjdG9yXG4gICk6IG51bWJlciB7XG4gICAgY29uc3QgY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kID1cbiAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbCB8fFxuICAgICAgaGFuZFZlY3RvckIucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ID09PSBudWxsXG4gICAgICAgID8gdW5kZWZpbmVkXG4gICAgICAgIDoge1xuICAgICAgICAgICAgLy8g5Y+z5omLIC0g6Kaq5oyHXG4gICAgICAgICAgICByaWdodFRodW1iVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgICAgICAgIHJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOS4reaMh1xuICAgICAgICAgICAgcmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5Y+z5omLIC0g6Jas5oyHXG4gICAgICAgICAgICByaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICAgICAgICAgcmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgIH07XG5cbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCA9XG4gICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbCB8fFxuICAgICAgaGFuZFZlY3RvckIubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICAgPyB1bmRlZmluZWRcbiAgICAgICAgOiB7XG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAgICAgICAgIGxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludDogUG9zZVNldC5nZXRDb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIGxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOS6uuW3ruOBl+aMh1xuICAgICAgICAgICAgbGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5Lit5oyHXG4gICAgICAgICAgICBsZWZ0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDolqzmjIdcbiAgICAgICAgICAgIGxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIGxlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOWwj+aMh1xuICAgICAgICAgICAgbGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50OiBQb3NlU2V0LmdldENvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFBvc2VTZXQuZ2V0Q29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgIH07XG5cbiAgICAvLyDlt6bmiYvjga7poZ7kvLzluqZcbiAgICBsZXQgY29zU2ltaWxhcml0aWVzU3VtTGVmdEhhbmQgPSAwO1xuICAgIGlmIChjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCkge1xuICAgICAgY29zU2ltaWxhcml0aWVzU3VtTGVmdEhhbmQgPSBPYmplY3QudmFsdWVzKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZFxuICAgICAgKS5yZWR1Y2UoKHN1bSwgdmFsdWUpID0+IHN1bSArIHZhbHVlLCAwKTtcbiAgICB9XG5cbiAgICAvLyDlj7PmiYvjga7poZ7kvLzluqZcbiAgICBsZXQgY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kID0gMDtcbiAgICBpZiAoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kKSB7XG4gICAgICBjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgPSBPYmplY3QudmFsdWVzKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmRcbiAgICAgICkucmVkdWNlKChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSwgMCk7XG4gICAgfVxuXG4gICAgLy8g5ZCI566X44GV44KM44Gf6aGe5Ly85bqmXG4gICAgaWYgKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCAmJiBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCkge1xuICAgICAgcmV0dXJuIChcbiAgICAgICAgKGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCArIGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kKSAvXG4gICAgICAgIChPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQhKS5sZW5ndGggK1xuICAgICAgICAgIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kISkubGVuZ3RoKVxuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCkge1xuICAgICAgaWYgKFxuICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCAhPT0gbnVsbCAmJlxuICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbFxuICAgICAgKSB7XG4gICAgICAgIC8vIGhhbmRWZWN0b3JCIOOBp+W3puaJi+OBjOOBguOCi+OBruOBqyBoYW5kVmVjdG9yQSDjgaflt6bmiYvjgYzjgarjgYTloLTlkIjjgIHpoZ7kvLzluqbjgpLmuJvjgonjgZlcbiAgICAgICAgY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgICBgW1Bvc2VTZXRdIGdldEhhbmRTaW1pbGFyaXR5IC0gQWRqdXN0IHNpbWlsYXJpdHksIGJlY2F1c2UgbGVmdCBoYW5kIG5vdCBmb3VuZC4uLmBcbiAgICAgICAgKTtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICBjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgL1xuICAgICAgICAgIChPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQhKS5sZW5ndGggKiAyKVxuICAgICAgICApO1xuICAgICAgfVxuICAgICAgcmV0dXJuIChcbiAgICAgICAgY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kIC9cbiAgICAgICAgT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kISkubGVuZ3RoXG4gICAgICApO1xuICAgIH0gZWxzZSBpZiAoY29zU2ltaWxhcml0aWVzTGVmdEhhbmQpIHtcbiAgICAgIGlmIChcbiAgICAgICAgaGFuZFZlY3RvckIucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ICE9PSBudWxsICYmXG4gICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbFxuICAgICAgKSB7XG4gICAgICAgIC8vIGhhbmRWZWN0b3JCIOOBp+WPs+aJi+OBjOOBguOCi+OBruOBqyBoYW5kVmVjdG9yQSDjgaflj7PmiYvjgYzjgarjgYTloLTlkIjjgIHpoZ7kvLzluqbjgpLmuJvjgonjgZlcbiAgICAgICAgY29uc29sZS5kZWJ1ZyhcbiAgICAgICAgICBgW1Bvc2VTZXRdIGdldEhhbmRTaW1pbGFyaXR5IC0gQWRqdXN0IHNpbWlsYXJpdHksIGJlY2F1c2UgcmlnaHQgaGFuZCBub3QgZm91bmQuLi5gXG4gICAgICAgICk7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgY29zU2ltaWxhcml0aWVzU3VtTGVmdEhhbmQgL1xuICAgICAgICAgIChPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCEpLmxlbmd0aCAqIDIpXG4gICAgICAgICk7XG4gICAgICB9XG4gICAgICByZXR1cm4gKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCAvXG4gICAgICAgIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kISkubGVuZ3RoXG4gICAgICApO1xuICAgIH1cblxuICAgIHJldHVybiAtMTtcbiAgfVxuXG4gIC8qKlxuICAgKiBaSVAg44OV44Kh44Kk44Or44Go44GX44Gm44Gu44K344Oq44Ki44Op44Kk44K6XG4gICAqIEByZXR1cm5zIFpJUOODleOCoeOCpOODqyAoQmxvYiDlvaLlvI8pXG4gICAqL1xuICBwdWJsaWMgYXN5bmMgZ2V0WmlwKCk6IFByb21pc2U8QmxvYj4ge1xuICAgIGNvbnN0IGpzWmlwID0gbmV3IEpTWmlwKCk7XG4gICAganNaaXAuZmlsZSgncG9zZXMuanNvbicsIGF3YWl0IHRoaXMuZ2V0SnNvbigpKTtcblxuICAgIGNvbnN0IGltYWdlRmlsZUV4dCA9IHRoaXMuZ2V0RmlsZUV4dGVuc2lvbkJ5TWltZSh0aGlzLklNQUdFX01JTUUpO1xuXG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGlmIChwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgY29uc3QgaW5kZXggPVxuICAgICAgICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UuZnJhbWVJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcbiAgICAgICAgICBqc1ppcC5maWxlKGBmcmFtZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ltYWdlRmlsZUV4dH1gLCBiYXNlNjQsIHtcbiAgICAgICAgICAgIGJhc2U2NDogdHJ1ZSxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgW1Bvc2VFeHBvcnRlclNlcnZpY2VdIHB1c2ggLSBDb3VsZCBub3QgcHVzaCBmcmFtZSBpbWFnZWAsXG4gICAgICAgICAgICBlcnJvclxuICAgICAgICAgICk7XG4gICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGlmIChwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICBjb25zdCBpbmRleCA9XG4gICAgICAgICAgICBwb3NlLnBvc2VJbWFnZURhdGFVcmwuaW5kZXhPZignYmFzZTY0LCcpICsgJ2Jhc2U2NCwnLmxlbmd0aDtcbiAgICAgICAgICBjb25zdCBiYXNlNjQgPSBwb3NlLnBvc2VJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcbiAgICAgICAgICBqc1ppcC5maWxlKGBwb3NlLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7aW1hZ2VGaWxlRXh0fWAsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZyYW1lIGltYWdlYCxcbiAgICAgICAgICAgIGVycm9yXG4gICAgICAgICAgKTtcbiAgICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgaWYgKHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgY29uc3QgaW5kZXggPVxuICAgICAgICAgICAgcG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwuaW5kZXhPZignYmFzZTY0LCcpICsgJ2Jhc2U2NCwnLmxlbmd0aDtcbiAgICAgICAgICBjb25zdCBiYXNlNjQgPSBwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYGZhY2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtpbWFnZUZpbGVFeHR9YCwgYmFzZTY0LCB7XG4gICAgICAgICAgICBiYXNlNjQ6IHRydWUsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlRXhwb3J0ZXJTZXJ2aWNlXSBwdXNoIC0gQ291bGQgbm90IHB1c2ggZmFjZSBmcmFtZSBpbWFnZWAsXG4gICAgICAgICAgICBlcnJvclxuICAgICAgICAgICk7XG4gICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gYXdhaXQganNaaXAuZ2VuZXJhdGVBc3luYyh7IHR5cGU6ICdibG9iJyB9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBKU09OIOaWh+Wtl+WIl+OBqOOBl+OBpuOBruOCt+ODquOCouODqeOCpOOCulxuICAgKiBAcmV0dXJucyBKU09OIOaWh+Wtl+WIl1xuICAgKi9cbiAgcHVibGljIGFzeW5jIGdldEpzb24oKTogUHJvbWlzZTxzdHJpbmc+IHtcbiAgICBpZiAodGhpcy52aWRlb01ldGFkYXRhID09PSB1bmRlZmluZWQgfHwgdGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKVxuICAgICAgcmV0dXJuICd7fSc7XG5cbiAgICBpZiAoIXRoaXMuaXNGaW5hbGl6ZWQpIHtcbiAgICAgIGF3YWl0IHRoaXMuZmluYWxpemUoKTtcbiAgICB9XG5cbiAgICBsZXQgcG9zZUxhbmRtYXJrTWFwcGluZ3MgPSBbXTtcbiAgICBmb3IgKGNvbnN0IGtleSBvZiBPYmplY3Qua2V5cyhQT1NFX0xBTkRNQVJLUykpIHtcbiAgICAgIGNvbnN0IGluZGV4OiBudW1iZXIgPSBQT1NFX0xBTkRNQVJLU1trZXkgYXMga2V5b2YgdHlwZW9mIFBPU0VfTEFORE1BUktTXTtcbiAgICAgIHBvc2VMYW5kbWFya01hcHBpbmdzW2luZGV4XSA9IGtleTtcbiAgICB9XG5cbiAgICBjb25zdCBqc29uOiBQb3NlU2V0SnNvbiA9IHtcbiAgICAgIGdlbmVyYXRvcjogJ21wLXZpZGVvLXBvc2UtZXh0cmFjdG9yJyxcbiAgICAgIHZlcnNpb246IDEsXG4gICAgICB2aWRlbzogdGhpcy52aWRlb01ldGFkYXRhISxcbiAgICAgIHBvc2VzOiB0aGlzLnBvc2VzLm1hcCgocG9zZTogUG9zZVNldEl0ZW0pOiBQb3NlU2V0SnNvbkl0ZW0gPT4ge1xuICAgICAgICAvLyBCb2R5VmVjdG9yIOOBruWcp+e4rlxuICAgICAgICBjb25zdCBib2R5VmVjdG9yID0gW107XG4gICAgICAgIGZvciAoY29uc3Qga2V5IG9mIFBvc2VTZXQuQk9EWV9WRUNUT1JfTUFQUElOR1MpIHtcbiAgICAgICAgICBib2R5VmVjdG9yLnB1c2gocG9zZS5ib2R5VmVjdG9yW2tleSBhcyBrZXlvZiBCb2R5VmVjdG9yXSk7XG4gICAgICAgIH1cblxuICAgICAgICAvLyBIYW5kVmVjdG9yIOOBruWcp+e4rlxuICAgICAgICBsZXQgaGFuZFZlY3RvcjogKG51bWJlcltdIHwgbnVsbClbXSB8IHVuZGVmaW5lZCA9IHVuZGVmaW5lZDtcbiAgICAgICAgaWYgKHBvc2UuaGFuZFZlY3Rvcikge1xuICAgICAgICAgIGhhbmRWZWN0b3IgPSBbXTtcbiAgICAgICAgICBmb3IgKGNvbnN0IGtleSBvZiBQb3NlU2V0LkhBTkRfVkVDVE9SX01BUFBJTkdTKSB7XG4gICAgICAgICAgICBoYW5kVmVjdG9yLnB1c2gocG9zZS5oYW5kVmVjdG9yW2tleSBhcyBrZXlvZiBIYW5kVmVjdG9yXSk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG5cbiAgICAgICAgLy8gUG9zZVNldEpzb25JdGVtIOOBriBwb3NlIOOCquODluOCuOOCp+OCr+ODiOOCkueUn+aIkFxuICAgICAgICByZXR1cm4ge1xuICAgICAgICAgIGlkOiBwb3NlLmlkLFxuICAgICAgICAgIHQ6IHBvc2UudGltZU1pbGlzZWNvbmRzLFxuICAgICAgICAgIGQ6IHBvc2UuZHVyYXRpb25NaWxpc2Vjb25kcyxcbiAgICAgICAgICBwOiBwb3NlLnBvc2UsXG4gICAgICAgICAgbDogcG9zZS5sZWZ0SGFuZCxcbiAgICAgICAgICByOiBwb3NlLnJpZ2h0SGFuZCxcbiAgICAgICAgICB2OiBib2R5VmVjdG9yLFxuICAgICAgICAgIGg6IGhhbmRWZWN0b3IsXG4gICAgICAgICAgZTogcG9zZS5leHRlbmRlZERhdGEsXG4gICAgICAgICAgbWQ6IHBvc2UubWVyZ2VkRHVyYXRpb25NaWxpc2Vjb25kcyxcbiAgICAgICAgICBtdDogcG9zZS5tZXJnZWRUaW1lTWlsaXNlY29uZHMsXG4gICAgICAgIH07XG4gICAgICB9KSxcbiAgICAgIHBvc2VMYW5kbWFya01hcHBwaW5nczogcG9zZUxhbmRtYXJrTWFwcGluZ3MsXG4gICAgfTtcblxuICAgIHJldHVybiBKU09OLnN0cmluZ2lmeShqc29uKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBKU09OIOOBi+OCieOBruiqreOBv+i+vOOBv1xuICAgKiBAcGFyYW0ganNvbiBKU09OIOaWh+Wtl+WIlyDjgb7jgZ/jga8gSlNPTiDjgqrjg5bjgrjjgqfjgq/jg4hcbiAgICovXG4gIGxvYWRKc29uKGpzb246IHN0cmluZyB8IGFueSkge1xuICAgIGNvbnN0IHBhcnNlZEpzb24gPSB0eXBlb2YganNvbiA9PT0gJ3N0cmluZycgPyBKU09OLnBhcnNlKGpzb24pIDoganNvbjtcblxuICAgIGlmIChwYXJzZWRKc29uLmdlbmVyYXRvciAhPT0gJ21wLXZpZGVvLXBvc2UtZXh0cmFjdG9yJykge1xuICAgICAgdGhyb3cgJ+S4jeato+OBquODleOCoeOCpOODqyc7XG4gICAgfSBlbHNlIGlmIChwYXJzZWRKc29uLnZlcnNpb24gIT09IDEpIHtcbiAgICAgIHRocm93ICfmnKrlr77lv5zjga7jg5Djg7zjgrjjg6fjg7MnO1xuICAgIH1cblxuICAgIHRoaXMudmlkZW9NZXRhZGF0YSA9IHBhcnNlZEpzb24udmlkZW87XG4gICAgdGhpcy5wb3NlcyA9IHBhcnNlZEpzb24ucG9zZXMubWFwKChpdGVtOiBQb3NlU2V0SnNvbkl0ZW0pOiBQb3NlU2V0SXRlbSA9PiB7XG4gICAgICBjb25zdCBib2R5VmVjdG9yOiBhbnkgPSB7fTtcbiAgICAgIFBvc2VTZXQuQk9EWV9WRUNUT1JfTUFQUElOR1MubWFwKChrZXksIGluZGV4KSA9PiB7XG4gICAgICAgIGJvZHlWZWN0b3Jba2V5IGFzIGtleW9mIEJvZHlWZWN0b3JdID0gaXRlbS52W2luZGV4XTtcbiAgICAgIH0pO1xuXG4gICAgICBjb25zdCBoYW5kVmVjdG9yOiBhbnkgPSB7fTtcbiAgICAgIGlmIChpdGVtLmgpIHtcbiAgICAgICAgUG9zZVNldC5IQU5EX1ZFQ1RPUl9NQVBQSU5HUy5tYXAoKGtleSwgaW5kZXgpID0+IHtcbiAgICAgICAgICBoYW5kVmVjdG9yW2tleSBhcyBrZXlvZiBIYW5kVmVjdG9yXSA9IGl0ZW0uaCFbaW5kZXhdO1xuICAgICAgICB9KTtcbiAgICAgIH1cblxuICAgICAgcmV0dXJuIHtcbiAgICAgICAgaWQ6XG4gICAgICAgICAgaXRlbS5pZCA9PT0gdW5kZWZpbmVkXG4gICAgICAgICAgICA/IFBvc2VTZXQuZ2V0SWRCeVRpbWVNaWxpc2Vjb25kcyhpdGVtLnQpXG4gICAgICAgICAgICA6IGl0ZW0uaWQsXG4gICAgICAgIHRpbWVNaWxpc2Vjb25kczogaXRlbS50LFxuICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBpdGVtLmQsXG4gICAgICAgIHBvc2U6IGl0ZW0ucCxcbiAgICAgICAgbGVmdEhhbmQ6IGl0ZW0ubCxcbiAgICAgICAgcmlnaHRIYW5kOiBpdGVtLnIsXG4gICAgICAgIGJvZHlWZWN0b3I6IGJvZHlWZWN0b3IsXG4gICAgICAgIGhhbmRWZWN0b3I6IGhhbmRWZWN0b3IsXG4gICAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiB1bmRlZmluZWQsXG4gICAgICAgIGV4dGVuZGVkRGF0YTogaXRlbS5lLFxuICAgICAgICBkZWJ1ZzogdW5kZWZpbmVkLFxuICAgICAgICBtZXJnZWREdXJhdGlvbk1pbGlzZWNvbmRzOiBpdGVtLm1kLFxuICAgICAgICBtZXJnZWRUaW1lTWlsaXNlY29uZHM6IGl0ZW0ubXQsXG4gICAgICB9O1xuICAgIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIFpJUCDjg5XjgqHjgqTjg6vjgYvjgonjga7oqq3jgb/ovrzjgb9cbiAgICogQHBhcmFtIGJ1ZmZlciBaSVAg44OV44Kh44Kk44Or44GuIEJ1ZmZlclxuICAgKiBAcGFyYW0gaW5jbHVkZUltYWdlcyDnlLvlg4/jgpLlsZXplovjgZnjgovjgYvjganjgYbjgYtcbiAgICovXG4gIGFzeW5jIGxvYWRaaXAoYnVmZmVyOiBBcnJheUJ1ZmZlciwgaW5jbHVkZUltYWdlczogYm9vbGVhbiA9IHRydWUpIHtcbiAgICBjb25zdCBqc1ppcCA9IG5ldyBKU1ppcCgpO1xuICAgIGNvbnNvbGUuZGVidWcoYFtQb3NlU2V0XSBpbml0Li4uYCk7XG4gICAgY29uc3QgemlwID0gYXdhaXQganNaaXAubG9hZEFzeW5jKGJ1ZmZlciwgeyBiYXNlNjQ6IGZhbHNlIH0pO1xuICAgIGlmICghemlwKSB0aHJvdyAnWklQ44OV44Kh44Kk44Or44KS6Kqt44G/6L6844KB44G+44Gb44KT44Gn44GX44GfJztcblxuICAgIGNvbnN0IGpzb24gPSBhd2FpdCB6aXAuZmlsZSgncG9zZXMuanNvbicpPy5hc3luYygndGV4dCcpO1xuICAgIGlmIChqc29uID09PSB1bmRlZmluZWQpIHtcbiAgICAgIHRocm93ICdaSVDjg5XjgqHjgqTjg6vjgasgcG9zZS5qc29uIOOBjOWQq+OBvuOCjOOBpuOBhOOBvuOBm+OCkyc7XG4gICAgfVxuXG4gICAgdGhpcy5sb2FkSnNvbihqc29uKTtcblxuICAgIGNvbnN0IGZpbGVFeHQgPSB0aGlzLmdldEZpbGVFeHRlbnNpb25CeU1pbWUodGhpcy5JTUFHRV9NSU1FKTtcblxuICAgIGlmIChpbmNsdWRlSW1hZ2VzKSB7XG4gICAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgICBjb25zdCBmcmFtZUltYWdlRmlsZU5hbWUgPSBgZnJhbWUtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtmaWxlRXh0fWA7XG4gICAgICAgICAgY29uc3QgaW1hZ2VCYXNlNjQgPSBhd2FpdCB6aXBcbiAgICAgICAgICAgIC5maWxlKGZyYW1lSW1hZ2VGaWxlTmFtZSlcbiAgICAgICAgICAgID8uYXN5bmMoJ2Jhc2U2NCcpO1xuICAgICAgICAgIGlmIChpbWFnZUJhc2U2NCkge1xuICAgICAgICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybCA9IGBkYXRhOiR7dGhpcy5JTUFHRV9NSU1FfTtiYXNlNjQsJHtpbWFnZUJhc2U2NH1gO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBpZiAoIXBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICAgIGNvbnN0IHBvc2VJbWFnZUZpbGVOYW1lID0gYHBvc2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtmaWxlRXh0fWA7XG4gICAgICAgICAgY29uc3QgaW1hZ2VCYXNlNjQgPSBhd2FpdCB6aXBcbiAgICAgICAgICAgIC5maWxlKHBvc2VJbWFnZUZpbGVOYW1lKVxuICAgICAgICAgICAgPy5hc3luYygnYmFzZTY0Jyk7XG4gICAgICAgICAgaWYgKGltYWdlQmFzZTY0KSB7XG4gICAgICAgICAgICBwb3NlLnBvc2VJbWFnZURhdGFVcmwgPSBgZGF0YToke3RoaXMuSU1BR0VfTUlNRX07YmFzZTY0LCR7aW1hZ2VCYXNlNjR9YDtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICBzdGF0aWMgZ2V0Q29zU2ltaWxhcml0eShhOiBudW1iZXJbXSwgYjogbnVtYmVyW10pIHtcbiAgICBpZiAoY29zU2ltaWxhcml0eUEpIHtcbiAgICAgIHJldHVybiBjb3NTaW1pbGFyaXR5QShhLCBiKTtcbiAgICB9XG4gICAgcmV0dXJuIGNvc1NpbWlsYXJpdHlCKGEsIGIpO1xuICB9XG5cbiAgcHJpdmF0ZSBwdXNoUG9zZUZyb21TaW1pbGFyUG9zZVF1ZXVlKG5leHRQb3NlVGltZU1pbGlzZWNvbmRzPzogbnVtYmVyKSB7XG4gICAgaWYgKHRoaXMuc2ltaWxhclBvc2VRdWV1ZS5sZW5ndGggPT09IDApIHJldHVybjtcblxuICAgIGlmICh0aGlzLnNpbWlsYXJQb3NlUXVldWUubGVuZ3RoID09PSAxKSB7XG4gICAgICAvLyDpoZ7kvLzjg53jg7zjgrrjgq3jg6Xjg7zjgavjg53jg7zjgrrjgYzkuIDjgaTjgZfjgYvjgarjgYTloLTlkIjjgIHlvZPoqbLjg53jg7zjgrrjgpLjg53jg7zjgrrphY3liJfjgbjov73liqBcbiAgICAgIGNvbnN0IHBvc2UgPSB0aGlzLnNpbWlsYXJQb3NlUXVldWVbMF07XG4gICAgICB0aGlzLnBvc2VzLnB1c2gocG9zZSk7XG4gICAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWUgPSBbXTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICAvLyDlkITjg53jg7zjgrrjga7mjIHntprmmYLplpPjgpLoqK3lrppcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMuc2ltaWxhclBvc2VRdWV1ZS5sZW5ndGggLSAxOyBpKyspIHtcbiAgICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZVtpXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlW2kgKyAxXS50aW1lTWlsaXNlY29uZHMgLVxuICAgICAgICB0aGlzLnNpbWlsYXJQb3NlUXVldWVbaV0udGltZU1pbGlzZWNvbmRzO1xuICAgIH1cbiAgICBpZiAobmV4dFBvc2VUaW1lTWlsaXNlY29uZHMpIHtcbiAgICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZVtcbiAgICAgICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlLmxlbmd0aCAtIDFcbiAgICAgIF0uZHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgIG5leHRQb3NlVGltZU1pbGlzZWNvbmRzIC1cbiAgICAgICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlW3RoaXMuc2ltaWxhclBvc2VRdWV1ZS5sZW5ndGggLSAxXS50aW1lTWlsaXNlY29uZHM7XG4gICAgfVxuXG4gICAgLy8g6aGe5Ly844Od44O844K644Kt44Ol44O844Gu5Lit44GL44KJ5pyA44KC5oyB57aa5pmC6ZaT44GM6ZW344GE44Od44O844K644KS6YG45oqeXG4gICAgY29uc3Qgc2VsZWN0ZWRQb3NlID0gUG9zZVNldC5nZXRTdWl0YWJsZVBvc2VCeVBvc2VzKHRoaXMuc2ltaWxhclBvc2VRdWV1ZSk7XG5cbiAgICAvLyDpgbjmip7jgZXjgozjgarjgYvjgaPjgZ/jg53jg7zjgrrjgpLliJfmjJlcbiAgICBzZWxlY3RlZFBvc2UuZGVidWcuZHVwbGljYXRlZEl0ZW1zID0gdGhpcy5zaW1pbGFyUG9zZVF1ZXVlXG4gICAgICAuZmlsdGVyKChpdGVtOiBQb3NlU2V0SXRlbSkgPT4ge1xuICAgICAgICByZXR1cm4gaXRlbS50aW1lTWlsaXNlY29uZHMgIT09IHNlbGVjdGVkUG9zZS50aW1lTWlsaXNlY29uZHM7XG4gICAgICB9KVxuICAgICAgLm1hcCgoaXRlbTogUG9zZVNldEl0ZW0pID0+IHtcbiAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICBpZDogaXRlbS5pZCxcbiAgICAgICAgICB0aW1lTWlsaXNlY29uZHM6IGl0ZW0udGltZU1pbGlzZWNvbmRzLFxuICAgICAgICAgIGR1cmF0aW9uTWlsaXNlY29uZHM6IGl0ZW0uZHVyYXRpb25NaWxpc2Vjb25kcyxcbiAgICAgICAgfTtcbiAgICAgIH0pO1xuICAgIHNlbGVjdGVkUG9zZS5tZXJnZWRUaW1lTWlsaXNlY29uZHMgPVxuICAgICAgdGhpcy5zaW1pbGFyUG9zZVF1ZXVlWzBdLnRpbWVNaWxpc2Vjb25kcztcbiAgICBzZWxlY3RlZFBvc2UubWVyZ2VkRHVyYXRpb25NaWxpc2Vjb25kcyA9IHRoaXMuc2ltaWxhclBvc2VRdWV1ZS5yZWR1Y2UoXG4gICAgICAoc3VtOiBudW1iZXIsIGl0ZW06IFBvc2VTZXRJdGVtKSA9PiB7XG4gICAgICAgIHJldHVybiBzdW0gKyBpdGVtLmR1cmF0aW9uTWlsaXNlY29uZHM7XG4gICAgICB9LFxuICAgICAgMFxuICAgICk7XG5cbiAgICAvLyDlvZPoqbLjg53jg7zjgrrjgpLjg53jg7zjgrrphY3liJfjgbjov73liqBcbiAgICBpZiAodGhpcy5JU19FTkFCTEVEX1JFTU9WRV9EVVBMSUNBVEVEX1BPU0VTX0ZPUl9BUk9VTkQpIHtcbiAgICAgIHRoaXMucG9zZXMucHVzaChzZWxlY3RlZFBvc2UpO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyDjg4fjg5Djg4PjgrDnlKhcbiAgICAgIHRoaXMucG9zZXMucHVzaCguLi50aGlzLnNpbWlsYXJQb3NlUXVldWUpO1xuICAgIH1cblxuICAgIC8vIOmhnuS8vOODneODvOOCuuOCreODpeODvOOCkuOCr+ODquOColxuICAgIHRoaXMuc2ltaWxhclBvc2VRdWV1ZSA9IFtdO1xuICB9XG5cbiAgcmVtb3ZlRHVwbGljYXRlZFBvc2VzKCk6IHZvaWQge1xuICAgIC8vIOWFqOODneODvOOCuuOCkuavlOi8g+OBl+OBpumhnuS8vOODneODvOOCuuOCkuWJiumZpFxuICAgIGNvbnN0IG5ld1Bvc2VzOiBQb3NlU2V0SXRlbVtdID0gW10sXG4gICAgICByZW1vdmVkUG9zZXM6IFBvc2VTZXRJdGVtW10gPSBbXTtcbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgbGV0IGR1cGxpY2F0ZWRQb3NlOiBQb3NlU2V0SXRlbTtcbiAgICAgIGZvciAoY29uc3QgaW5zZXJ0ZWRQb3NlIG9mIG5ld1Bvc2VzKSB7XG4gICAgICAgIGNvbnN0IGlzU2ltaWxhckJvZHlQb3NlID0gUG9zZVNldC5pc1NpbWlsYXJCb2R5UG9zZShcbiAgICAgICAgICBwb3NlLmJvZHlWZWN0b3IsXG4gICAgICAgICAgaW5zZXJ0ZWRQb3NlLmJvZHlWZWN0b3JcbiAgICAgICAgKTtcbiAgICAgICAgY29uc3QgaXNTaW1pbGFySGFuZFBvc2UgPVxuICAgICAgICAgIHBvc2UuaGFuZFZlY3RvciAmJiBpbnNlcnRlZFBvc2UuaGFuZFZlY3RvclxuICAgICAgICAgICAgPyBQb3NlU2V0LmlzU2ltaWxhckhhbmRQb3NlKFxuICAgICAgICAgICAgICAgIHBvc2UuaGFuZFZlY3RvcixcbiAgICAgICAgICAgICAgICBpbnNlcnRlZFBvc2UuaGFuZFZlY3RvcixcbiAgICAgICAgICAgICAgICAwLjlcbiAgICAgICAgICAgICAgKVxuICAgICAgICAgICAgOiBmYWxzZTtcblxuICAgICAgICBpZiAoaXNTaW1pbGFyQm9keVBvc2UgJiYgaXNTaW1pbGFySGFuZFBvc2UpIHtcbiAgICAgICAgICAvLyDouqvkvZPjg7vmiYvjgajjgoLjgavpoZ7kvLzjg53jg7zjgrrjgarjgonjgbBcbiAgICAgICAgICBkdXBsaWNhdGVkUG9zZSA9IGluc2VydGVkUG9zZTtcbiAgICAgICAgICBicmVhaztcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICBpZiAoZHVwbGljYXRlZFBvc2UpIHtcbiAgICAgICAgcmVtb3ZlZFBvc2VzLnB1c2gocG9zZSk7XG4gICAgICAgIGlmIChkdXBsaWNhdGVkUG9zZS5kZWJ1Zy5kdXBsaWNhdGVkSXRlbXMpIHtcbiAgICAgICAgICBkdXBsaWNhdGVkUG9zZS5kZWJ1Zy5kdXBsaWNhdGVkSXRlbXMucHVzaCh7XG4gICAgICAgICAgICBpZDogcG9zZS5pZCxcbiAgICAgICAgICAgIHRpbWVNaWxpc2Vjb25kczogcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBwb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG5cbiAgICAgIG5ld1Bvc2VzLnB1c2gocG9zZSk7XG4gICAgfVxuXG4gICAgY29uc29sZS5pbmZvKFxuICAgICAgYFtQb3NlU2V0XSByZW1vdmVEdXBsaWNhdGVkUG9zZXMgLSBSZWR1Y2VkICR7dGhpcy5wb3Nlcy5sZW5ndGh9IHBvc2VzIC0+ICR7bmV3UG9zZXMubGVuZ3RofSBwb3Nlc2AsXG4gICAgICB7XG4gICAgICAgIHJlbW92ZWQ6IHJlbW92ZWRQb3NlcyxcbiAgICAgICAga2VlcGVkOiBuZXdQb3NlcyxcbiAgICAgIH1cbiAgICApO1xuICAgIHRoaXMucG9zZXMgPSBuZXdQb3NlcztcbiAgfVxuXG4gIHN0YXRpYyBnZXRJZEJ5VGltZU1pbGlzZWNvbmRzKHRpbWVNaWxpc2Vjb25kczogbnVtYmVyKSB7XG4gICAgcmV0dXJuIE1hdGguZmxvb3IodGltZU1pbGlzZWNvbmRzIC8gMTAwKSAqIDEwMDtcbiAgfVxuXG4gIHByaXZhdGUgZ2V0RmlsZUV4dGVuc2lvbkJ5TWltZShJTUFHRV9NSU1FOiBzdHJpbmcpIHtcbiAgICBzd2l0Y2ggKElNQUdFX01JTUUpIHtcbiAgICAgIGNhc2UgJ2ltYWdlL3BuZyc6XG4gICAgICAgIHJldHVybiAncG5nJztcbiAgICAgIGNhc2UgJ2ltYWdlL2pwZWcnOlxuICAgICAgICByZXR1cm4gJ2pwZyc7XG4gICAgICBjYXNlICdpbWFnZS93ZWJwJzpcbiAgICAgICAgcmV0dXJuICd3ZWJwJztcbiAgICAgIGRlZmF1bHQ6XG4gICAgICAgIHJldHVybiAncG5nJztcbiAgICB9XG4gIH1cbn1cbiJdfQ==