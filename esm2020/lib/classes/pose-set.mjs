import { POSE_LANDMARKS } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
// @ts-ignore
import cosSimilarity from 'cos-similarity';
import { ImageTrimmer } from './internals/image-trimmer';
export class PoseSet {
    constructor() {
        this.poses = [];
        this.isFinalized = false;
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
    getNumberOfPoses() {
        if (this.poses === undefined)
            return -1;
        return this.poses.length;
    }
    getPoses() {
        if (this.poses === undefined)
            return [];
        return this.poses;
    }
    getPoseByTime(timeMiliseconds) {
        if (this.poses === undefined)
            return undefined;
        return this.poses.find((pose) => pose.timeMiliseconds === timeMiliseconds);
    }
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
        const handVector = PoseSet.getHandVectors(results.leftHandLandmarks, results.rightHandLandmarks);
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
            rightHand: results.leftHandLandmarks?.map((normalizedLandmark) => {
                return [
                    normalizedLandmark.x,
                    normalizedLandmark.y,
                    normalizedLandmark.z,
                ];
            }),
            bodyVectors: bodyVector,
            handVectors: handVector,
            frameImageDataUrl: frameImageDataUrl,
            poseImageDataUrl: poseImageDataUrl,
            faceFrameImageDataUrl: faceFrameImageDataUrl,
            extendedData: {},
        };
        if (1 <= this.poses.length) {
            // 前回のポーズとの類似性をチェック
            const lastPose = this.poses[this.poses.length - 1];
            const isSimilarBodyPose = PoseSet.isSimilarBodyPose(lastPose.bodyVectors, pose.bodyVectors);
            let isSimilarHandPose = true;
            if (lastPose.handVectors && pose.handVectors) {
                isSimilarHandPose = PoseSet.isSimilarHandPose(lastPose.handVectors, pose.handVectors);
            }
            else if (!lastPose.handVectors && pose.handVectors) {
                isSimilarHandPose = false;
            }
            if (isSimilarBodyPose && isSimilarHandPose) {
                // 身体・手ともに類似ポーズならばスキップ
                return;
            }
            // 前回のポーズの持続時間を設定
            const poseDurationMiliseconds = videoTimeMiliseconds - lastPose.timeMiliseconds;
            this.poses[this.poses.length - 1].durationMiliseconds =
                poseDurationMiliseconds;
        }
        this.poses.push(pose);
        return pose;
    }
    async finalize() {
        if (0 == this.poses.length) {
            this.isFinalized = true;
            return;
        }
        // 最後のポーズの持続時間を設定
        if (1 <= this.poses.length) {
            const lastPose = this.poses[this.poses.length - 1];
            if (lastPose.durationMiliseconds == -1) {
                const poseDurationMiliseconds = this.videoMetadata.duration - lastPose.timeMiliseconds;
                this.poses[this.poses.length - 1].durationMiliseconds =
                    poseDurationMiliseconds;
            }
        }
        // 重複ポーズを除去
        this.removeDuplicatedPoses();
        // 最初のポーズを除去
        this.poses.shift();
        // 画像のマージンを取得
        console.log(`[PoseSet] finalize - Detecting image margins...`);
        let imageTrimming = undefined;
        for (const pose of this.poses) {
            let imageTrimmer = new ImageTrimmer();
            if (!pose.frameImageDataUrl) {
                continue;
            }
            await imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);
            const marginColor = await imageTrimmer.getMarginColor();
            console.log(`[PoseSet] finalize - Detected margin color...`, pose.timeMiliseconds, marginColor);
            if (marginColor === null)
                continue;
            if (marginColor !== this.IMAGE_MARGIN_TRIMMING_COLOR) {
                continue;
            }
            const trimmed = await imageTrimmer.trimMargin(marginColor, this.IMAGE_MARGIN_TRIMMING_DIFF_THRESHOLD);
            if (!trimmed)
                continue;
            imageTrimming = trimmed;
            console.log(`[PoseSet] finalize - Determined image trimming positions...`, trimmed);
            break;
        }
        // 画像を整形
        for (const pose of this.poses) {
            let imageTrimmer = new ImageTrimmer();
            if (!pose.frameImageDataUrl || !pose.poseImageDataUrl) {
                continue;
            }
            console.log(`[PoseSet] finalize - Processing image...`, pose.timeMiliseconds);
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
    removeDuplicatedPoses() {
        // 全ポーズを比較して類似ポーズを削除
        const newPoses = [];
        for (const poseA of this.poses) {
            let isDuplicated = false;
            for (const poseB of newPoses) {
                const isSimilarBodyPose = PoseSet.isSimilarBodyPose(poseA.bodyVectors, poseB.bodyVectors);
                const isSimilarHandPose = poseA.handVectors && poseB.handVectors
                    ? PoseSet.isSimilarHandPose(poseA.handVectors, poseB.handVectors)
                    : false;
                if (isSimilarBodyPose && isSimilarHandPose) {
                    // 身体・手ともに類似ポーズならば
                    isDuplicated = true;
                    break;
                }
            }
            if (isDuplicated)
                continue;
            newPoses.push(poseA);
        }
        console.info(`[PoseSet] removeDuplicatedPoses - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`);
        this.poses = newPoses;
    }
    getSimilarPoses(results, threshold = 0.9, targetRange = 'all') {
        // 身体のベクトルを取得
        let bodyVector = PoseSet.getBodyVector(results.ea);
        if (!bodyVector) {
            throw 'Could not get the body vector';
        }
        // 手指のベクトルを取得
        let handVector;
        if (targetRange === 'all' || targetRange === 'handPose') {
            handVector = PoseSet.getHandVectors(results.leftHandLandmarks, results.rightHandLandmarks);
            if (targetRange === 'handPose' && !handVector) {
                throw 'Could not get the hand vector';
            }
        }
        // 各ポーズとベクトルを比較
        const poses = [];
        for (const pose of this.poses) {
            if ((targetRange === 'all' || targetRange === 'bodyPose') &&
                !pose.bodyVectors) {
                continue;
            }
            else if (targetRange === 'handPose' && !pose.handVectors) {
                continue;
            }
            // 身体のポーズの類似度を取得
            let bodySimilarity;
            if (bodyVector && pose.bodyVectors) {
                bodySimilarity = PoseSet.getBodyPoseSimilarity(pose.bodyVectors, bodyVector);
            }
            // 手指のポーズの類似度を取得
            let handSimilarity;
            if (handVector && pose.handVectors) {
                handSimilarity = PoseSet.getHandSimilarity(pose.handVectors, handVector);
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
    static getHandVectors(leftHandLandmarks, rightHandLandmarks) {
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
    static isSimilarBodyPose(bodyVectorA, bodyVectorB, threshold = 0.8) {
        let isSimilar = false;
        const similarity = PoseSet.getBodyPoseSimilarity(bodyVectorA, bodyVectorB);
        if (similarity >= threshold)
            isSimilar = true;
        // console.log(`[PoseSet] isSimilarPose`, isSimilar, similarity);
        return isSimilar;
    }
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
    static isSimilarHandPose(handVectorA, handVectorB, threshold = 0.75) {
        const similarity = PoseSet.getHandSimilarity(handVectorA, handVectorB);
        if (similarity === -1) {
            return true;
        }
        return similarity >= threshold;
    }
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
        let cosSimilaritiesSumLeftHand = 0;
        if (cosSimilaritiesLeftHand) {
            cosSimilaritiesSumLeftHand = Object.values(cosSimilaritiesLeftHand).reduce((sum, value) => sum + value, 0);
        }
        let cosSimilaritiesSumRightHand = 0;
        if (cosSimilaritiesRightHand) {
            cosSimilaritiesSumRightHand = Object.values(cosSimilaritiesRightHand).reduce((sum, value) => sum + value, 0);
        }
        if (cosSimilaritiesRightHand && cosSimilaritiesLeftHand) {
            return ((cosSimilaritiesSumRightHand + cosSimilaritiesSumLeftHand) /
                (Object.keys(cosSimilaritiesRightHand).length +
                    Object.keys(cosSimilaritiesLeftHand).length));
        }
        else if (cosSimilaritiesSumRightHand) {
            return (cosSimilaritiesSumRightHand /
                Object.keys(cosSimilaritiesRightHand).length);
        }
        else if (cosSimilaritiesLeftHand) {
            return (cosSimilaritiesSumLeftHand /
                Object.keys(cosSimilaritiesLeftHand).length);
        }
        else {
            return -1;
        }
    }
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
                    bodyVector.push(pose.bodyVectors[key]);
                }
                // HandVector の圧縮
                let handVector = undefined;
                if (pose.handVectors) {
                    handVector = [];
                    for (const key of PoseSet.HAND_VECTOR_MAPPINGS) {
                        handVector.push(pose.handVectors[key]);
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
                };
            }),
            poseLandmarkMapppings: poseLandmarkMappings,
        };
        return JSON.stringify(json);
    }
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
                bodyVectors: bodyVector,
                handVectors: handVector,
                frameImageDataUrl: undefined,
                extendedData: item.e,
            };
        });
    }
    async loadZip(buffer, includeImages = true) {
        console.log(`[PoseSet] loadZip...`, JSZip);
        const jsZip = new JSZip();
        console.log(`[PoseSet] init...`);
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
    // 右足
    'rightAnkleToRightKnee',
    'rightKneeToRightHip',
    // 左足
    'leftAnkleToLeftKnee',
    'leftKneeToLeftHip',
    // 胴体
    'rightHipToLeftHip',
    'rightShoulderToLeftShoulder',
];
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1zZXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi9wcm9qZWN0cy9uZ3gtbXAtcG9zZS1leHRyYWN0b3Ivc3JjL2xpYi9jbGFzc2VzL3Bvc2Utc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxjQUFjLEVBQVcsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssS0FBSyxNQUFNLE9BQU8sQ0FBQztBQU0vQixhQUFhO0FBQ2IsT0FBTyxhQUFhLE1BQU0sZ0JBQWdCLENBQUM7QUFFM0MsT0FBTyxFQUFFLFlBQVksRUFBRSxNQUFNLDJCQUEyQixDQUFDO0FBR3pELE1BQU0sT0FBTyxPQUFPO0lBaUZsQjtRQXZFTyxVQUFLLEdBQWtCLEVBQUUsQ0FBQztRQUMxQixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQXVEckMsYUFBYTtRQUNJLGdCQUFXLEdBQVcsSUFBSSxDQUFDO1FBQzNCLGVBQVUsR0FDekIsWUFBWSxDQUFDO1FBQ0Usa0JBQWEsR0FBRyxHQUFHLENBQUM7UUFFckMsVUFBVTtRQUNPLGdDQUEyQixHQUFHLFNBQVMsQ0FBQztRQUN4Qyx5Q0FBb0MsR0FBRyxFQUFFLENBQUM7UUFFM0QsV0FBVztRQUNNLHVDQUFrQyxHQUFHLFNBQVMsQ0FBQztRQUMvQyx1Q0FBa0MsR0FBRyxXQUFXLENBQUM7UUFDakQsNENBQXVDLEdBQUcsR0FBRyxDQUFDO1FBRzdELElBQUksQ0FBQyxhQUFhLEdBQUc7WUFDbkIsSUFBSSxFQUFFLEVBQUU7WUFDUixLQUFLLEVBQUUsQ0FBQztZQUNSLE1BQU0sRUFBRSxDQUFDO1lBQ1QsUUFBUSxFQUFFLENBQUM7WUFDWCxxQkFBcUIsRUFBRSxDQUFDO1NBQ3pCLENBQUM7SUFDSixDQUFDO0lBRUQsWUFBWTtRQUNWLE9BQU8sSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUM7SUFDakMsQ0FBQztJQUVELFlBQVksQ0FBQyxTQUFpQjtRQUM1QixJQUFJ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R0FBRyxRQUFRLElBQUksQ0FBQyxVQUFVLFdBQVcsV0FBVyxFQUFFLENBQUM7cUJBQ3pFO2lCQUNGO2FBQ0Y7U0FDRjtJQUNILENBQUM7O0FBN25DRCxrQkFBa0I7QUFDSyw0QkFBb0IsR0FBRztJQUM1QyxLQUFLO0lBQ0wsd0JBQXdCO0lBQ3hCLDJCQUEyQjtJQUMzQixLQUFLO0lBQ0wsc0JBQXNCO0lBQ3RCLHlCQUF5QjtDQUMxQixDQUFDO0FBRUYsa0JBQWtCO0FBQ0ssNEJBQW9CLEdBQUc7SUFDNUMsVUFBVTtJQUNWLDJCQUEyQjtJQUMzQixtQ0FBbUM7SUFDbkMsWUFBWTtJQUNaLGlDQUFpQztJQUNqQyx5Q0FBeUM7SUFDekMsVUFBVTtJQUNWLGtDQUFrQztJQUNsQywwQ0FBMEM7SUFDMUMsVUFBVTtJQUNWLGdDQUFnQztJQUNoQyx3Q0FBd0M7SUFDeEMsVUFBVTtJQUNWLGlDQUFpQztJQUNqQyx5Q0FBeUM7SUFDekMsVUFBVTtJQUNWLDBCQUEwQjtJQUMxQixrQ0FBa0M7SUFDbEMsWUFBWTtJQUNaLGdDQUFnQztJQUNoQyx3Q0FBd0M7SUFDeEMsVUFBVTtJQUNWLGlDQUFpQztJQUNqQyx5Q0FBeUM7SUFDekMsVUFBVTtJQUNWLCtCQUErQjtJQUMvQix1Q0FBdUM7SUFDdkMsVUFBVTtJQUNWLGdDQUFnQztJQUNoQyx3Q0FBd0M7SUFDeEMsS0FBSztJQUNMLHVCQUF1QjtJQUN2QixxQkFBcUI7SUFDckIsS0FBSztJQUNMLHFCQUFxQjtJQUNyQixtQkFBbUI7SUFDbkIsS0FBSztJQUNMLG1CQUFtQjtJQUNuQiw2QkFBNkI7Q0FDOUIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IFBPU0VfTEFORE1BUktTLCBSZXN1bHRzIH0gZnJvbSAnQG1lZGlhcGlwZS9ob2xpc3RpYyc7XG5pbXBvcnQgKiBhcyBKU1ppcCBmcm9tICdqc3ppcCc7XG5pbXBvcnQgeyBQb3NlU2V0SXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1zZXQtaXRlbSc7XG5pbXBvcnQgeyBQb3NlU2V0SnNvbiB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1zZXQtanNvbic7XG5pbXBvcnQgeyBQb3NlU2V0SnNvbkl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWpzb24taXRlbSc7XG5pbXBvcnQgeyBCb2R5VmVjdG9yIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9ib2R5LXZlY3Rvcic7XG5cbi8vIEB0cy1pZ25vcmVcbmltcG9ydCBjb3NTaW1pbGFyaXR5IGZyb20gJ2Nvcy1zaW1pbGFyaXR5JztcbmltcG9ydCB7IFNpbWlsYXJQb3NlSXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvc2ltaWxhci1wb3NlLWl0ZW0nO1xuaW1wb3J0IHsgSW1hZ2VUcmltbWVyIH0gZnJvbSAnLi9pbnRlcm5hbHMvaW1hZ2UtdHJpbW1lcic7XG5pbXBvcnQgeyBIYW5kVmVjdG9yIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9oYW5kLXZlY3Rvcic7XG5cbmV4cG9ydCBjbGFzcyBQb3NlU2V0IHtcbiAgcHVibGljIGdlbmVyYXRvcj86IHN0cmluZztcbiAgcHVibGljIHZlcnNpb24/OiBudW1iZXI7XG4gIHByaXZhdGUgdmlkZW9NZXRhZGF0YSE6IHtcbiAgICBuYW1lOiBzdHJpbmc7XG4gICAgd2lkdGg6IG51bWJlcjtcbiAgICBoZWlnaHQ6IG51bWJlcjtcbiAgICBkdXJhdGlvbjogbnVtYmVyO1xuICAgIGZpcnN0UG9zZURldGVjdGVkVGltZTogbnVtYmVyO1xuICB9O1xuICBwdWJsaWMgcG9zZXM6IFBvc2VTZXRJdGVtW10gPSBbXTtcbiAgcHVibGljIGlzRmluYWxpemVkPzogYm9vbGVhbiA9IGZhbHNlO1xuXG4gIC8vIEJvZHlWZWN0b3Ig44Gu44Kt44O85ZCNXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgQk9EWV9WRUNUT1JfTUFQUElOR1MgPSBbXG4gICAgLy8g5Y+z6IWVXG4gICAgJ3JpZ2h0V3Jpc3RUb1JpZ2h0RWxib3cnLFxuICAgICdyaWdodEVsYm93VG9SaWdodFNob3VsZGVyJyxcbiAgICAvLyDlt6bohZVcbiAgICAnbGVmdFdyaXN0VG9MZWZ0RWxib3cnLFxuICAgICdsZWZ0RWxib3dUb0xlZnRTaG91bGRlcicsXG4gIF07XG5cbiAgLy8gSGFuZFZlY3RvciDjga7jgq3jg7zlkI1cbiAgcHVibGljIHN0YXRpYyByZWFkb25seSBIQU5EX1ZFQ1RPUl9NQVBQSU5HUyA9IFtcbiAgICAvLyDlj7PmiYsgLSDopqrmjIdcbiAgICAncmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ3JpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5Y+z5omLIC0g5Lq65beu44GX5oyHXG4gICAgJ3JpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOS4reaMh1xuICAgICdyaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ3JpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOiWrOaMh1xuICAgICdyaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5Y+z5omLIC0g5bCP5oyHXG4gICAgJ3JpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOimquaMh1xuICAgICdsZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g5Lq65beu44GX5oyHXG4gICAgJ2xlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlt6bmiYsgLSDkuK3mjIdcbiAgICAnbGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g6Jas5oyHXG4gICAgJ2xlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAnbGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g5bCP5oyHXG4gICAgJ2xlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PotrNcbiAgICAncmlnaHRBbmtsZVRvUmlnaHRLbmVlJyxcbiAgICAncmlnaHRLbmVlVG9SaWdodEhpcCcsXG4gICAgLy8g5bem6LazXG4gICAgJ2xlZnRBbmtsZVRvTGVmdEtuZWUnLFxuICAgICdsZWZ0S25lZVRvTGVmdEhpcCcsXG4gICAgLy8g6IO05L2TXG4gICAgJ3JpZ2h0SGlwVG9MZWZ0SGlwJyxcbiAgICAncmlnaHRTaG91bGRlclRvTGVmdFNob3VsZGVyJyxcbiAgXTtcblxuICAvLyDnlLvlg4/mm7jjgY3lh7rjgZfmmYLjga7oqK3lrppcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9XSURUSDogbnVtYmVyID0gMTA4MDtcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9NSU1FOiAnaW1hZ2UvanBlZycgfCAnaW1hZ2UvcG5nJyB8ICdpbWFnZS93ZWJwJyA9XG4gICAgJ2ltYWdlL3dlYnAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX1FVQUxJVFkgPSAwLjg7XG5cbiAgLy8g55S75YOP44Gu5L2Z55m96Zmk5Y67XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0NPTE9SID0gJyMwMDAwMDAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX01BUkdJTl9UUklNTUlOR19ESUZGX1RIUkVTSE9MRCA9IDUwO1xuXG4gIC8vIOeUu+WDj+OBruiDjOaZr+iJsue9ruaPm1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9TUkNfQ09MT1IgPSAnIzAxNkFGRCc7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RTVF9DT0xPUiA9ICcjRkZGRkZGMDAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9ESUZGX1RIUkVTSE9MRCA9IDEzMDtcblxuICBjb25zdHJ1Y3RvcigpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSB7XG4gICAgICBuYW1lOiAnJyxcbiAgICAgIHdpZHRoOiAwLFxuICAgICAgaGVpZ2h0OiAwLFxuICAgICAgZHVyYXRpb246IDAsXG4gICAgICBmaXJzdFBvc2VEZXRlY3RlZFRpbWU6IDAsXG4gICAgfTtcbiAgfVxuXG4gIGdldFZpZGVvTmFtZSgpIHtcbiAgICByZXR1cm4gdGhpcy52aWRlb01ldGFkYXRhLm5hbWU7XG4gIH1cblxuICBzZXRWaWRlb05hbWUodmlkZW9OYW1lOiBzdHJpbmcpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEubmFtZSA9IHZpZGVvTmFtZTtcbiAgfVxuXG4gIHNldFZpZGVvTWV0YURhdGEod2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIsIGR1cmF0aW9uOiBudW1iZXIpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEud2lkdGggPSB3aWR0aDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuaGVpZ2h0ID0gaGVpZ2h0O1xuICAgIHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbiA9IGR1cmF0aW9uO1xuICB9XG5cbiAgZ2V0TnVtYmVyT2ZQb3NlcygpOiBudW1iZXIge1xuICAgIGlmICh0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpIHJldHVybiAtMTtcbiAgICByZXR1cm4gdGhpcy5wb3Nlcy5sZW5ndGg7XG4gIH1cblxuICBnZXRQb3NlcygpOiBQb3NlU2V0SXRlbVtdIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gW107XG4gICAgcmV0dXJuIHRoaXMucG9zZXM7XG4gIH1cblxuICBnZXRQb3NlQnlUaW1lKHRpbWVNaWxpc2Vjb25kczogbnVtYmVyKTogUG9zZVNldEl0ZW0gfCB1bmRlZmluZWQge1xuICAgIGlmICh0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpIHJldHVybiB1bmRlZmluZWQ7XG4gICAgcmV0dXJuIHRoaXMucG9zZXMuZmluZCgocG9zZSkgPT4gcG9zZS50aW1lTWlsaXNlY29uZHMgPT09IHRpbWVNaWxpc2Vjb25kcyk7XG4gIH1cblxuICBwdXNoUG9zZShcbiAgICB2aWRlb1RpbWVNaWxpc2Vjb25kczogbnVtYmVyLFxuICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gICAgcG9zZUltYWdlRGF0YVVybDogc3RyaW5nIHwgdW5kZWZpbmVkLFxuICAgIGZhY2VGcmFtZUltYWdlRGF0YVVybDogc3RyaW5nIHwgdW5kZWZpbmVkLFxuICAgIHJlc3VsdHM6IFJlc3VsdHNcbiAgKTogUG9zZVNldEl0ZW0gfCB1bmRlZmluZWQge1xuICAgIGlmIChyZXN1bHRzLnBvc2VMYW5kbWFya3MgPT09IHVuZGVmaW5lZCkgcmV0dXJuO1xuXG4gICAgaWYgKHRoaXMucG9zZXMubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZmlyc3RQb3NlRGV0ZWN0ZWRUaW1lID0gdmlkZW9UaW1lTWlsaXNlY29uZHM7XG4gICAgfVxuXG4gICAgY29uc3QgcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGU6IGFueVtdID0gKHJlc3VsdHMgYXMgYW55KS5lYVxuICAgICAgPyAocmVzdWx0cyBhcyBhbnkpLmVhXG4gICAgICA6IFtdO1xuICAgIGlmIChwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZS5sZW5ndGggPT09IDApIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgcG9zZSB3aXRoIHRoZSB3b3JsZCBjb29yZGluYXRlYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICBjb25zdCBib2R5VmVjdG9yID0gUG9zZVNldC5nZXRCb2R5VmVjdG9yKHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlKTtcbiAgICBpZiAoIWJvZHlWZWN0b3IpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgYm9keSB2ZWN0b3JgLFxuICAgICAgICBwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZVxuICAgICAgKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICBpZiAoXG4gICAgICByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgJiZcbiAgICAgIHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWRcbiAgICApIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgaGFuZCBsYW5kbWFya3NgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH0gZWxzZSBpZiAocmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIGxlZnQgaGFuZCBsYW5kbWFya3NgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH0gZWxzZSBpZiAocmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSByaWdodCBoYW5kIGxhbmRtYXJrc2AsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgfVxuXG4gICAgY29uc3QgaGFuZFZlY3RvciA9IFBvc2VTZXQuZ2V0SGFuZFZlY3RvcnMoXG4gICAgICByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzLFxuICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3NcbiAgICApO1xuICAgIGlmICghaGFuZFZlY3Rvcikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBoYW5kIHZlY3RvcmAsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgfVxuXG4gICAgY29uc3QgcG9zZTogUG9zZVNldEl0ZW0gPSB7XG4gICAgICB0aW1lTWlsaXNlY29uZHM6IHZpZGVvVGltZU1pbGlzZWNvbmRzLFxuICAgICAgZHVyYXRpb25NaWxpc2Vjb25kczogLTEsXG4gICAgICBwb3NlOiBwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZS5tYXAoKHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsueCxcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay55LFxuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLnosXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsudmlzaWJpbGl0eSxcbiAgICAgICAgXTtcbiAgICAgIH0pLFxuICAgICAgbGVmdEhhbmQ6IHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3M/Lm1hcCgobm9ybWFsaXplZExhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLngsXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnksXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnosXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIHJpZ2h0SGFuZDogcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcz8ubWFwKChub3JtYWxpemVkTGFuZG1hcmspID0+IHtcbiAgICAgICAgcmV0dXJuIFtcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueCxcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueSxcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueixcbiAgICAgICAgXTtcbiAgICAgIH0pLFxuICAgICAgYm9keVZlY3RvcnM6IGJvZHlWZWN0b3IsXG4gICAgICBoYW5kVmVjdG9yczogaGFuZFZlY3RvcixcbiAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiBmcmFtZUltYWdlRGF0YVVybCxcbiAgICAgIHBvc2VJbWFnZURhdGFVcmw6IHBvc2VJbWFnZURhdGFVcmwsXG4gICAgICBmYWNlRnJhbWVJbWFnZURhdGFVcmw6IGZhY2VGcmFtZUltYWdlRGF0YVVybCxcbiAgICAgIGV4dGVuZGVkRGF0YToge30sXG4gICAgfTtcblxuICAgIGlmICgxIDw9IHRoaXMucG9zZXMubGVuZ3RoKSB7XG4gICAgICAvLyDliY3lm57jga7jg53jg7zjgrrjgajjga7poZ7kvLzmgKfjgpLjg4Hjgqfjg4Pjgq9cbiAgICAgIGNvbnN0IGxhc3RQb3NlID0gdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdO1xuXG4gICAgICBjb25zdCBpc1NpbWlsYXJCb2R5UG9zZSA9IFBvc2VTZXQuaXNTaW1pbGFyQm9keVBvc2UoXG4gICAgICAgIGxhc3RQb3NlLmJvZHlWZWN0b3JzLFxuICAgICAgICBwb3NlLmJvZHlWZWN0b3JzXG4gICAgICApO1xuXG4gICAgICBsZXQgaXNTaW1pbGFySGFuZFBvc2UgPSB0cnVlO1xuICAgICAgaWYgKGxhc3RQb3NlLmhhbmRWZWN0b3JzICYmIHBvc2UuaGFuZFZlY3RvcnMpIHtcbiAgICAgICAgaXNTaW1pbGFySGFuZFBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckhhbmRQb3NlKFxuICAgICAgICAgIGxhc3RQb3NlLmhhbmRWZWN0b3JzLFxuICAgICAgICAgIHBvc2UuaGFuZFZlY3RvcnNcbiAgICAgICAgKTtcbiAgICAgIH0gZWxzZSBpZiAoIWxhc3RQb3NlLmhhbmRWZWN0b3JzICYmIHBvc2UuaGFuZFZlY3RvcnMpIHtcbiAgICAgICAgaXNTaW1pbGFySGFuZFBvc2UgPSBmYWxzZTtcbiAgICAgIH1cblxuICAgICAgaWYgKGlzU2ltaWxhckJvZHlQb3NlICYmIGlzU2ltaWxhckhhbmRQb3NlKSB7XG4gICAgICAgIC8vIOi6q+S9k+ODu+aJi+OBqOOCguOBq+mhnuS8vOODneODvOOCuuOBquOCieOBsOOCueOCreODg+ODl1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG5cbiAgICAgIC8vIOWJjeWbnuOBruODneODvOOCuuOBruaMgee2muaZgumWk+OCkuioreWumlxuICAgICAgY29uc3QgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICB2aWRlb1RpbWVNaWxpc2Vjb25kcyAtIGxhc3RQb3NlLnRpbWVNaWxpc2Vjb25kcztcbiAgICAgIHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHM7XG4gICAgfVxuXG4gICAgdGhpcy5wb3Nlcy5wdXNoKHBvc2UpO1xuXG4gICAgcmV0dXJuIHBvc2U7XG4gIH1cblxuICBhc3luYyBmaW5hbGl6ZSgpIHtcbiAgICBpZiAoMCA9PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgdGhpcy5pc0ZpbmFsaXplZCA9IHRydWU7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgLy8g5pyA5b6M44Gu44Od44O844K644Gu5oyB57aa5pmC6ZaT44KS6Kit5a6aXG4gICAgaWYgKDEgPD0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIGNvbnN0IGxhc3RQb3NlID0gdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdO1xuICAgICAgaWYgKGxhc3RQb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMgPT0gLTEpIHtcbiAgICAgICAgY29uc3QgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICAgIHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbiAtIGxhc3RQb3NlLnRpbWVNaWxpc2Vjb25kcztcbiAgICAgICAgdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICAgIHBvc2VEdXJhdGlvbk1pbGlzZWNvbmRzO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIOmHjeikh+ODneODvOOCuuOCkumZpOWOu1xuICAgIHRoaXMucmVtb3ZlRHVwbGljYXRlZFBvc2VzKCk7XG5cbiAgICAvLyDmnIDliJ3jga7jg53jg7zjgrrjgpLpmaTljrtcbiAgICB0aGlzLnBvc2VzLnNoaWZ0KCk7XG5cbiAgICAvLyDnlLvlg4/jga7jg57jg7zjgrjjg7PjgpLlj5blvpdcbiAgICBjb25zb2xlLmxvZyhgW1Bvc2VTZXRdIGZpbmFsaXplIC0gRGV0ZWN0aW5nIGltYWdlIG1hcmdpbnMuLi5gKTtcbiAgICBsZXQgaW1hZ2VUcmltbWluZzpcbiAgICAgIHwge1xuICAgICAgICAgIG1hcmdpblRvcDogbnVtYmVyO1xuICAgICAgICAgIG1hcmdpbkJvdHRvbTogbnVtYmVyO1xuICAgICAgICAgIGhlaWdodE5ldzogbnVtYmVyO1xuICAgICAgICAgIGhlaWdodE9sZDogbnVtYmVyO1xuICAgICAgICAgIHdpZHRoOiBudW1iZXI7XG4gICAgICAgIH1cbiAgICAgIHwgdW5kZWZpbmVkID0gdW5kZWZpbmVkO1xuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5mcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGNvbnN0IG1hcmdpbkNvbG9yID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldE1hcmdpbkNvbG9yKCk7XG4gICAgICBjb25zb2xlLmxvZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVjdGVkIG1hcmdpbiBjb2xvci4uLmAsXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzLFxuICAgICAgICBtYXJnaW5Db2xvclxuICAgICAgKTtcbiAgICAgIGlmIChtYXJnaW5Db2xvciA9PT0gbnVsbCkgY29udGludWU7XG4gICAgICBpZiAobWFyZ2luQ29sb3IgIT09IHRoaXMuSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0NPTE9SKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgY29uc3QgdHJpbW1lZCA9IGF3YWl0IGltYWdlVHJpbW1lci50cmltTWFyZ2luKFxuICAgICAgICBtYXJnaW5Db2xvcixcbiAgICAgICAgdGhpcy5JTUFHRV9NQVJHSU5fVFJJTU1JTkdfRElGRl9USFJFU0hPTERcbiAgICAgICk7XG4gICAgICBpZiAoIXRyaW1tZWQpIGNvbnRpbnVlO1xuICAgICAgaW1hZ2VUcmltbWluZyA9IHRyaW1tZWQ7XG4gICAgICBjb25zb2xlLmxvZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVybWluZWQgaW1hZ2UgdHJpbW1pbmcgcG9zaXRpb25zLi4uYCxcbiAgICAgICAgdHJpbW1lZFxuICAgICAgKTtcbiAgICAgIGJyZWFrO1xuICAgIH1cblxuICAgIC8vIOeUu+WDj+OCkuaVtOW9olxuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsIHx8ICFwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG5cbiAgICAgIGNvbnNvbGUubG9nKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gUHJvY2Vzc2luZyBpbWFnZS4uLmAsXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzXG4gICAgICApO1xuXG4gICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDjg5Xjg6zjg7zjg6DnlLvlg49cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UuZnJhbWVJbWFnZURhdGFVcmwpO1xuXG4gICAgICBpZiAoaW1hZ2VUcmltbWluZykge1xuICAgICAgICBhd2FpdCBpbWFnZVRyaW1tZXIuY3JvcChcbiAgICAgICAgICAwLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcubWFyZ2luVG9wLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcud2lkdGgsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5oZWlnaHROZXdcbiAgICAgICAgKTtcbiAgICAgIH1cblxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLnJlcGxhY2VDb2xvcihcbiAgICAgICAgdGhpcy5JTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfU1JDX0NPTE9SLFxuICAgICAgICB0aGlzLklNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9EU1RfQ09MT1IsXG4gICAgICAgIHRoaXMuSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RJRkZfVEhSRVNIT0xEXG4gICAgICApO1xuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVzaXplV2l0aEZpdCh7XG4gICAgICAgIHdpZHRoOiB0aGlzLklNQUdFX1dJRFRILFxuICAgICAgfSk7XG5cbiAgICAgIGxldCBuZXdEYXRhVXJsID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldERhdGFVcmwoXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSxcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2UvanBlZycgfHwgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2Uvd2VicCdcbiAgICAgICAgICA/IHRoaXMuSU1BR0VfUVVBTElUWVxuICAgICAgICAgIDogdW5kZWZpbmVkXG4gICAgICApO1xuICAgICAgaWYgKCFuZXdEYXRhVXJsKSB7XG4gICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gQ291bGQgbm90IGdldCB0aGUgbmV3IGRhdGF1cmwgZm9yIGZyYW1lIGltYWdlYFxuICAgICAgICApO1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIHBvc2UuZnJhbWVJbWFnZURhdGFVcmwgPSBuZXdEYXRhVXJsO1xuXG4gICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDjg53jg7zjgrrjg5fjg6zjg5Pjg6Xjg7znlLvlg49cbiAgICAgIGltYWdlVHJpbW1lciA9IG5ldyBJbWFnZVRyaW1tZXIoKTtcbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UucG9zZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGlmIChpbWFnZVRyaW1taW5nKSB7XG4gICAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5jcm9wKFxuICAgICAgICAgIDAsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5tYXJnaW5Ub3AsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy53aWR0aCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLmhlaWdodE5ld1xuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVzaXplV2l0aEZpdCh7XG4gICAgICAgIHdpZHRoOiB0aGlzLklNQUdFX1dJRFRILFxuICAgICAgfSk7XG5cbiAgICAgIG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgID8gdGhpcy5JTUFHRV9RVUFMSVRZXG4gICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICk7XG4gICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgcG9zZSBwcmV2aWV3IGltYWdlYFxuICAgICAgICApO1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG5cbiAgICAgIGlmIChwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDpoZTjg5Xjg6zjg7zjg6DnlLvlg49cbiAgICAgICAgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgICAgbmV3RGF0YVVybCA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXREYXRhVXJsKFxuICAgICAgICAgIHRoaXMuSU1BR0VfTUlNRSxcbiAgICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgICAgPyB0aGlzLklNQUdFX1FVQUxJVFlcbiAgICAgICAgICAgIDogdW5kZWZpbmVkXG4gICAgICAgICk7XG4gICAgICAgIGlmICghbmV3RGF0YVVybCkge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgZmFjZSBmcmFtZSBpbWFnZWBcbiAgICAgICAgICApO1xuICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICB9XG4gICAgICAgIHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsID0gbmV3RGF0YVVybDtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB0aGlzLmlzRmluYWxpemVkID0gdHJ1ZTtcbiAgfVxuXG4gIHJlbW92ZUR1cGxpY2F0ZWRQb3NlcygpOiB2b2lkIHtcbiAgICAvLyDlhajjg53jg7zjgrrjgpLmr5TovIPjgZfjgabpoZ7kvLzjg53jg7zjgrrjgpLliYrpmaRcbiAgICBjb25zdCBuZXdQb3NlczogUG9zZVNldEl0ZW1bXSA9IFtdO1xuICAgIGZvciAoY29uc3QgcG9zZUEgb2YgdGhpcy5wb3Nlcykge1xuICAgICAgbGV0IGlzRHVwbGljYXRlZCA9IGZhbHNlO1xuICAgICAgZm9yIChjb25zdCBwb3NlQiBvZiBuZXdQb3Nlcykge1xuICAgICAgICBjb25zdCBpc1NpbWlsYXJCb2R5UG9zZSA9IFBvc2VTZXQuaXNTaW1pbGFyQm9keVBvc2UoXG4gICAgICAgICAgcG9zZUEuYm9keVZlY3RvcnMsXG4gICAgICAgICAgcG9zZUIuYm9keVZlY3RvcnNcbiAgICAgICAgKTtcbiAgICAgICAgY29uc3QgaXNTaW1pbGFySGFuZFBvc2UgPVxuICAgICAgICAgIHBvc2VBLmhhbmRWZWN0b3JzICYmIHBvc2VCLmhhbmRWZWN0b3JzXG4gICAgICAgICAgICA/IFBvc2VTZXQuaXNTaW1pbGFySGFuZFBvc2UocG9zZUEuaGFuZFZlY3RvcnMsIHBvc2VCLmhhbmRWZWN0b3JzKVxuICAgICAgICAgICAgOiBmYWxzZTtcblxuICAgICAgICBpZiAoaXNTaW1pbGFyQm9keVBvc2UgJiYgaXNTaW1pbGFySGFuZFBvc2UpIHtcbiAgICAgICAgICAvLyDouqvkvZPjg7vmiYvjgajjgoLjgavpoZ7kvLzjg53jg7zjgrrjgarjgonjgbBcbiAgICAgICAgICBpc0R1cGxpY2F0ZWQgPSB0cnVlO1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChpc0R1cGxpY2F0ZWQpIGNvbnRpbnVlO1xuXG4gICAgICBuZXdQb3Nlcy5wdXNoKHBvc2VBKTtcbiAgICB9XG5cbiAgICBjb25zb2xlLmluZm8oXG4gICAgICBgW1Bvc2VTZXRdIHJlbW92ZUR1cGxpY2F0ZWRQb3NlcyAtIFJlZHVjZWQgJHt0aGlzLnBvc2VzLmxlbmd0aH0gcG9zZXMgLT4gJHtuZXdQb3Nlcy5sZW5ndGh9IHBvc2VzYFxuICAgICk7XG4gICAgdGhpcy5wb3NlcyA9IG5ld1Bvc2VzO1xuICB9XG5cbiAgZ2V0U2ltaWxhclBvc2VzKFxuICAgIHJlc3VsdHM6IFJlc3VsdHMsXG4gICAgdGhyZXNob2xkOiBudW1iZXIgPSAwLjksXG4gICAgdGFyZ2V0UmFuZ2U6ICdhbGwnIHwgJ2JvZHlQb3NlJyB8ICdoYW5kUG9zZScgPSAnYWxsJ1xuICApOiBTaW1pbGFyUG9zZUl0ZW1bXSB7XG4gICAgLy8g6Lqr5L2T44Gu44OZ44Kv44OI44Or44KS5Y+W5b6XXG4gICAgbGV0IGJvZHlWZWN0b3I6IEJvZHlWZWN0b3IgPSBQb3NlU2V0LmdldEJvZHlWZWN0b3IoKHJlc3VsdHMgYXMgYW55KS5lYSk7XG4gICAgaWYgKCFib2R5VmVjdG9yKSB7XG4gICAgICB0aHJvdyAnQ291bGQgbm90IGdldCB0aGUgYm9keSB2ZWN0b3InO1xuICAgIH1cblxuICAgIC8vIOaJi+aMh+OBruODmeOCr+ODiOODq+OCkuWPluW+l1xuICAgIGxldCBoYW5kVmVjdG9yOiBIYW5kVmVjdG9yO1xuICAgIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2FsbCcgfHwgdGFyZ2V0UmFuZ2UgPT09ICdoYW5kUG9zZScpIHtcbiAgICAgIGhhbmRWZWN0b3IgPSBQb3NlU2V0LmdldEhhbmRWZWN0b3JzKFxuICAgICAgICByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzLFxuICAgICAgICByZXN1bHRzLnJpZ2h0SGFuZExhbmRtYXJrc1xuICAgICAgKTtcbiAgICAgIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2hhbmRQb3NlJyAmJiAhaGFuZFZlY3Rvcikge1xuICAgICAgICB0aHJvdyAnQ291bGQgbm90IGdldCB0aGUgaGFuZCB2ZWN0b3InO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIOWQhOODneODvOOCuuOBqOODmeOCr+ODiOODq+OCkuavlOi8g1xuICAgIGNvbnN0IHBvc2VzID0gW107XG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGlmIChcbiAgICAgICAgKHRhcmdldFJhbmdlID09PSAnYWxsJyB8fCB0YXJnZXRSYW5nZSA9PT0gJ2JvZHlQb3NlJykgJiZcbiAgICAgICAgIXBvc2UuYm9keVZlY3RvcnNcbiAgICAgICkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH0gZWxzZSBpZiAodGFyZ2V0UmFuZ2UgPT09ICdoYW5kUG9zZScgJiYgIXBvc2UuaGFuZFZlY3RvcnMpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG5cbiAgICAgIC8vIOi6q+S9k+OBruODneODvOOCuuOBrumhnuS8vOW6puOCkuWPluW+l1xuICAgICAgbGV0IGJvZHlTaW1pbGFyaXR5OiBudW1iZXI7XG4gICAgICBpZiAoYm9keVZlY3RvciAmJiBwb3NlLmJvZHlWZWN0b3JzKSB7XG4gICAgICAgIGJvZHlTaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRCb2R5UG9zZVNpbWlsYXJpdHkoXG4gICAgICAgICAgcG9zZS5ib2R5VmVjdG9ycyxcbiAgICAgICAgICBib2R5VmVjdG9yXG4gICAgICAgICk7XG4gICAgICB9XG5cbiAgICAgIC8vIOaJi+aMh+OBruODneODvOOCuuOBrumhnuS8vOW6puOCkuWPluW+l1xuICAgICAgbGV0IGhhbmRTaW1pbGFyaXR5OiBudW1iZXI7XG4gICAgICBpZiAoaGFuZFZlY3RvciAmJiBwb3NlLmhhbmRWZWN0b3JzKSB7XG4gICAgICAgIGhhbmRTaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRIYW5kU2ltaWxhcml0eShcbiAgICAgICAgICBwb3NlLmhhbmRWZWN0b3JzLFxuICAgICAgICAgIGhhbmRWZWN0b3JcbiAgICAgICAgKTtcbiAgICAgIH1cblxuICAgICAgLy8g5Yik5a6aXG4gICAgICBsZXQgc2ltaWxhcml0eTogbnVtYmVyLFxuICAgICAgICBpc1NpbWlsYXIgPSBmYWxzZTtcbiAgICAgIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2FsbCcpIHtcbiAgICAgICAgc2ltaWxhcml0eSA9IE1hdGgubWF4KGJvZHlTaW1pbGFyaXR5ID8/IDAsIGhhbmRTaW1pbGFyaXR5ID8/IDApO1xuICAgICAgICBpZiAodGhyZXNob2xkIDw9IGJvZHlTaW1pbGFyaXR5IHx8IHRocmVzaG9sZCA8PSBoYW5kU2ltaWxhcml0eSkge1xuICAgICAgICAgIGlzU2ltaWxhciA9IHRydWU7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSBpZiAodGFyZ2V0UmFuZ2UgPT09ICdib2R5UG9zZScpIHtcbiAgICAgICAgc2ltaWxhcml0eSA9IGJvZHlTaW1pbGFyaXR5O1xuICAgICAgICBpZiAodGhyZXNob2xkIDw9IGJvZHlTaW1pbGFyaXR5KSB7XG4gICAgICAgICAgaXNTaW1pbGFyID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2hhbmRQb3NlJykge1xuICAgICAgICBzaW1pbGFyaXR5ID0gaGFuZFNpbWlsYXJpdHk7XG4gICAgICAgIGlmICh0aHJlc2hvbGQgPD0gaGFuZFNpbWlsYXJpdHkpIHtcbiAgICAgICAgICBpc1NpbWlsYXIgPSB0cnVlO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmICghaXNTaW1pbGFyKSBjb250aW51ZTtcblxuICAgICAgLy8g57WQ5p6c44G46L+95YqgXG4gICAgICBwb3Nlcy5wdXNoKHtcbiAgICAgICAgLi4ucG9zZSxcbiAgICAgICAgc2ltaWxhcml0eTogc2ltaWxhcml0eSxcbiAgICAgICAgYm9keVBvc2VTaW1pbGFyaXR5OiBib2R5U2ltaWxhcml0eSxcbiAgICAgICAgaGFuZFBvc2VTaW1pbGFyaXR5OiBoYW5kU2ltaWxhcml0eSxcbiAgICAgIH0gYXMgU2ltaWxhclBvc2VJdGVtKTtcbiAgICB9XG5cbiAgICByZXR1cm4gcG9zZXM7XG4gIH1cblxuICBzdGF0aWMgZ2V0Qm9keVZlY3RvcihcbiAgICBwb3NlTGFuZG1hcmtzOiB7IHg6IG51bWJlcjsgeTogbnVtYmVyOyB6OiBudW1iZXIgfVtdXG4gICk6IEJvZHlWZWN0b3IgfCB1bmRlZmluZWQge1xuICAgIHJldHVybiB7XG4gICAgICByaWdodFdyaXN0VG9SaWdodEVsYm93OiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnosXG4gICAgICBdLFxuICAgICAgcmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcjogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS56LFxuICAgICAgXSxcbiAgICAgIGxlZnRXcmlzdFRvTGVmdEVsYm93OiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnosXG4gICAgICBdLFxuICAgICAgbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXI6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueixcbiAgICAgIF0sXG4gICAgfTtcbiAgfVxuXG4gIHN0YXRpYyBnZXRIYW5kVmVjdG9ycyhcbiAgICBsZWZ0SGFuZExhbmRtYXJrczogeyB4OiBudW1iZXI7IHk6IG51bWJlcjsgejogbnVtYmVyIH1bXSxcbiAgICByaWdodEhhbmRMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W11cbiAgKTogSGFuZFZlY3RvciB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKFxuICAgICAgKHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDApICYmXG4gICAgICAobGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDApXG4gICAgKSB7XG4gICAgICByZXR1cm4gdW5kZWZpbmVkO1xuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICAvLyDlj7PmiYsgLSDopqrmjIdcbiAgICAgIHJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnggLSByaWdodEhhbmRMYW5kbWFya3NbM10ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbM10ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnogLSByaWdodEhhbmRMYW5kbWFya3NbM10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgIHJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnggLSByaWdodEhhbmRMYW5kbWFya3NbN10ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbN10ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnogLSByaWdodEhhbmRMYW5kbWFya3NbN10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnggLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnkgLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnogLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDkuK3mjIdcbiAgICAgIHJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMl0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzEyXS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzExXS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTJdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTFdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTFdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTBdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMF0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzExXS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzEwXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOiWrOaMh1xuICAgICAgcmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNl0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE2XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTZdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTVdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE0XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTVdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTRdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDlsI/mjIdcbiAgICAgIHJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzIwXS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMjBdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTldLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1syMF0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE4XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTldLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMThdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAgIGxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbNF0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzNdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzRdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1szXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s0XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbM10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbM10ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzJdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1syXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1szXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgIGxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbOF0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzddLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzhdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1s3XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s4XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbN10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbN10ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzZdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzddLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1s2XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s3XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbNl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDkuK3mjIdcbiAgICAgIGxlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOiWrOaMh1xuICAgICAgbGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDlsI/mjIdcbiAgICAgIGxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgfTtcbiAgfVxuXG4gIHN0YXRpYyBpc1NpbWlsYXJCb2R5UG9zZShcbiAgICBib2R5VmVjdG9yQTogQm9keVZlY3RvcixcbiAgICBib2R5VmVjdG9yQjogQm9keVZlY3RvcixcbiAgICB0aHJlc2hvbGQgPSAwLjhcbiAgKTogYm9vbGVhbiB7XG4gICAgbGV0IGlzU2ltaWxhciA9IGZhbHNlO1xuICAgIGNvbnN0IHNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEJvZHlQb3NlU2ltaWxhcml0eShib2R5VmVjdG9yQSwgYm9keVZlY3RvckIpO1xuICAgIGlmIChzaW1pbGFyaXR5ID49IHRocmVzaG9sZCkgaXNTaW1pbGFyID0gdHJ1ZTtcblxuICAgIC8vIGNvbnNvbGUubG9nKGBbUG9zZVNldF0gaXNTaW1pbGFyUG9zZWAsIGlzU2ltaWxhciwgc2ltaWxhcml0eSk7XG5cbiAgICByZXR1cm4gaXNTaW1pbGFyO1xuICB9XG5cbiAgc3RhdGljIGdldEJvZHlQb3NlU2ltaWxhcml0eShcbiAgICBib2R5VmVjdG9yQTogQm9keVZlY3RvcixcbiAgICBib2R5VmVjdG9yQjogQm9keVZlY3RvclxuICApOiBudW1iZXIge1xuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllcyA9IHtcbiAgICAgIGxlZnRXcmlzdFRvTGVmdEVsYm93OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5sZWZ0V3Jpc3RUb0xlZnRFbGJvdyxcbiAgICAgICAgYm9keVZlY3RvckIubGVmdFdyaXN0VG9MZWZ0RWxib3dcbiAgICAgICksXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogY29zU2ltaWxhcml0eShcbiAgICAgICAgYm9keVZlY3RvckEubGVmdEVsYm93VG9MZWZ0U2hvdWxkZXIsXG4gICAgICAgIGJvZHlWZWN0b3JCLmxlZnRFbGJvd1RvTGVmdFNob3VsZGVyXG4gICAgICApLFxuICAgICAgcmlnaHRXcmlzdFRvUmlnaHRFbGJvdzogY29zU2ltaWxhcml0eShcbiAgICAgICAgYm9keVZlY3RvckEucmlnaHRXcmlzdFRvUmlnaHRFbGJvdyxcbiAgICAgICAgYm9keVZlY3RvckIucmlnaHRXcmlzdFRvUmlnaHRFbGJvd1xuICAgICAgKSxcbiAgICAgIHJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXI6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXIsXG4gICAgICAgIGJvZHlWZWN0b3JCLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXJcbiAgICAgICksXG4gICAgfTtcblxuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllc1N1bSA9IE9iamVjdC52YWx1ZXMoY29zU2ltaWxhcml0aWVzKS5yZWR1Y2UoXG4gICAgICAoc3VtLCB2YWx1ZSkgPT4gc3VtICsgdmFsdWUsXG4gICAgICAwXG4gICAgKTtcbiAgICByZXR1cm4gY29zU2ltaWxhcml0aWVzU3VtIC8gT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzKS5sZW5ndGg7XG4gIH1cblxuICBzdGF0aWMgaXNTaW1pbGFySGFuZFBvc2UoXG4gICAgaGFuZFZlY3RvckE6IEhhbmRWZWN0b3IsXG4gICAgaGFuZFZlY3RvckI6IEhhbmRWZWN0b3IsXG4gICAgdGhyZXNob2xkID0gMC43NVxuICApOiBib29sZWFuIHtcbiAgICBjb25zdCBzaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRIYW5kU2ltaWxhcml0eShoYW5kVmVjdG9yQSwgaGFuZFZlY3RvckIpO1xuICAgIGlmIChzaW1pbGFyaXR5ID09PSAtMSkge1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuICAgIHJldHVybiBzaW1pbGFyaXR5ID49IHRocmVzaG9sZDtcbiAgfVxuXG4gIHN0YXRpYyBnZXRIYW5kU2ltaWxhcml0eShcbiAgICBoYW5kVmVjdG9yQTogSGFuZFZlY3RvcixcbiAgICBoYW5kVmVjdG9yQjogSGFuZFZlY3RvclxuICApOiBudW1iZXIge1xuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCA9XG4gICAgICBoYW5kVmVjdG9yQS5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGwgfHxcbiAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbFxuICAgICAgICA/IHVuZGVmaW5lZFxuICAgICAgICA6IHtcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOimquaMh1xuICAgICAgICAgICAgcmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgICAgICAgIHJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5Y+z5omLIC0g5Lit5oyHXG4gICAgICAgICAgICByaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOiWrOaMh1xuICAgICAgICAgICAgcmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICAgICAgICAgcmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgfTtcbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCA9XG4gICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbCB8fFxuICAgICAgaGFuZFZlY3RvckIubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICAgPyB1bmRlZmluZWRcbiAgICAgICAgOiB7XG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAgICAgICAgIGxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5Lq65beu44GX5oyHXG4gICAgICAgICAgICBsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOS4reaMh1xuICAgICAgICAgICAgbGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIGxlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDolqzmjIdcbiAgICAgICAgICAgIGxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5bCP5oyHXG4gICAgICAgICAgICBsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICB9O1xuXG4gICAgbGV0IGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kID0gMDtcbiAgICBpZiAoY29zU2ltaWxhcml0aWVzTGVmdEhhbmQpIHtcbiAgICAgIGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kID0gT2JqZWN0LnZhbHVlcyhcbiAgICAgICAgY29zU2ltaWxhcml0aWVzTGVmdEhhbmRcbiAgICAgICkucmVkdWNlKChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSwgMCk7XG4gICAgfVxuXG4gICAgbGV0IGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCA9IDA7XG4gICAgaWYgKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCkge1xuICAgICAgY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kID0gT2JqZWN0LnZhbHVlcyhcbiAgICAgICAgY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kXG4gICAgICApLnJlZHVjZSgoc3VtLCB2YWx1ZSkgPT4gc3VtICsgdmFsdWUsIDApO1xuICAgIH1cblxuICAgIGlmIChjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQgJiYgY29zU2ltaWxhcml0aWVzTGVmdEhhbmQpIHtcbiAgICAgIHJldHVybiAoXG4gICAgICAgIChjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgKyBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCkgL1xuICAgICAgICAoT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kISkubGVuZ3RoICtcbiAgICAgICAgICBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCEpLmxlbmd0aClcbiAgICAgICk7XG4gICAgfSBlbHNlIGlmIChjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQpIHtcbiAgICAgIHJldHVybiAoXG4gICAgICAgIGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCAvXG4gICAgICAgIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCEpLmxlbmd0aFxuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kKSB7XG4gICAgICByZXR1cm4gKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCAvXG4gICAgICAgIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kISkubGVuZ3RoXG4gICAgICApO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gLTE7XG4gICAgfVxuICB9XG5cbiAgcHVibGljIGFzeW5jIGdldFppcCgpOiBQcm9taXNlPEJsb2I+IHtcbiAgICBjb25zdCBqc1ppcCA9IG5ldyBKU1ppcCgpO1xuICAgIGpzWmlwLmZpbGUoJ3Bvc2VzLmpzb24nLCBhd2FpdCB0aGlzLmdldEpzb24oKSk7XG5cbiAgICBjb25zdCBpbWFnZUZpbGVFeHQgPSB0aGlzLmdldEZpbGVFeHRlbnNpb25CeU1pbWUodGhpcy5JTUFHRV9NSU1FKTtcblxuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBpZiAocG9zZS5mcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UuZnJhbWVJbWFnZURhdGFVcmwuaW5kZXhPZignYmFzZTY0LCcpICsgJ2Jhc2U2NCwnLmxlbmd0aDtcbiAgICAgICAgICBjb25zdCBiYXNlNjQgPSBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsLnN1YnN0cmluZyhpbmRleCk7XG4gICAgICAgICAganNaaXAuZmlsZShgZnJhbWUtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtpbWFnZUZpbGVFeHR9YCwgYmFzZTY0LCB7XG4gICAgICAgICAgICBiYXNlNjQ6IHRydWUsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlRXhwb3J0ZXJTZXJ2aWNlXSBwdXNoIC0gQ291bGQgbm90IHB1c2ggZnJhbWUgaW1hZ2VgLFxuICAgICAgICAgICAgZXJyb3JcbiAgICAgICAgICApO1xuICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBpZiAocG9zZS5wb3NlSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgY29uc3QgaW5kZXggPVxuICAgICAgICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5wb3NlSW1hZ2VEYXRhVXJsLnN1YnN0cmluZyhpbmRleCk7XG4gICAgICAgICAganNaaXAuZmlsZShgcG9zZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ltYWdlRmlsZUV4dH1gLCBiYXNlNjQsIHtcbiAgICAgICAgICAgIGJhc2U2NDogdHJ1ZSxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgW1Bvc2VFeHBvcnRlclNlcnZpY2VdIHB1c2ggLSBDb3VsZCBub3QgcHVzaCBmcmFtZSBpbWFnZWAsXG4gICAgICAgICAgICBlcnJvclxuICAgICAgICAgICk7XG4gICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGlmIChwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcbiAgICAgICAgICBqc1ppcC5maWxlKGBmYWNlLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7aW1hZ2VGaWxlRXh0fWAsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZhY2UgZnJhbWUgaW1hZ2VgLFxuICAgICAgICAgICAgZXJyb3JcbiAgICAgICAgICApO1xuICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGF3YWl0IGpzWmlwLmdlbmVyYXRlQXN5bmMoeyB0eXBlOiAnYmxvYicgfSk7XG4gIH1cblxuICBnZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKElNQUdFX01JTUU6IHN0cmluZykge1xuICAgIHN3aXRjaCAoSU1BR0VfTUlNRSkge1xuICAgICAgY2FzZSAnaW1hZ2UvcG5nJzpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgICAgY2FzZSAnaW1hZ2UvanBlZyc6XG4gICAgICAgIHJldHVybiAnanBnJztcbiAgICAgIGNhc2UgJ2ltYWdlL3dlYnAnOlxuICAgICAgICByZXR1cm4gJ3dlYnAnO1xuICAgICAgZGVmYXVsdDpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgIH1cbiAgfVxuXG4gIHB1YmxpYyBhc3luYyBnZXRKc29uKCk6IFByb21pc2U8c3RyaW5nPiB7XG4gICAgaWYgKHRoaXMudmlkZW9NZXRhZGF0YSA9PT0gdW5kZWZpbmVkIHx8IHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZClcbiAgICAgIHJldHVybiAne30nO1xuXG4gICAgaWYgKCF0aGlzLmlzRmluYWxpemVkKSB7XG4gICAgICBhd2FpdCB0aGlzLmZpbmFsaXplKCk7XG4gICAgfVxuXG4gICAgbGV0IHBvc2VMYW5kbWFya01hcHBpbmdzID0gW107XG4gICAgZm9yIChjb25zdCBrZXkgb2YgT2JqZWN0LmtleXMoUE9TRV9MQU5ETUFSS1MpKSB7XG4gICAgICBjb25zdCBpbmRleDogbnVtYmVyID0gUE9TRV9MQU5ETUFSS1Nba2V5IGFzIGtleW9mIHR5cGVvZiBQT1NFX0xBTkRNQVJLU107XG4gICAgICBwb3NlTGFuZG1hcmtNYXBwaW5nc1tpbmRleF0gPSBrZXk7XG4gICAgfVxuXG4gICAgY29uc3QganNvbjogUG9zZVNldEpzb24gPSB7XG4gICAgICBnZW5lcmF0b3I6ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicsXG4gICAgICB2ZXJzaW9uOiAxLFxuICAgICAgdmlkZW86IHRoaXMudmlkZW9NZXRhZGF0YSEsXG4gICAgICBwb3NlczogdGhpcy5wb3Nlcy5tYXAoKHBvc2U6IFBvc2VTZXRJdGVtKTogUG9zZVNldEpzb25JdGVtID0+IHtcbiAgICAgICAgLy8gQm9keVZlY3RvciDjga7lnKfnuK5cbiAgICAgICAgY29uc3QgYm9keVZlY3RvciA9IFtdO1xuICAgICAgICBmb3IgKGNvbnN0IGtleSBvZiBQb3NlU2V0LkJPRFlfVkVDVE9SX01BUFBJTkdTKSB7XG4gICAgICAgICAgYm9keVZlY3Rvci5wdXNoKHBvc2UuYm9keVZlY3RvcnNba2V5IGFzIGtleW9mIEJvZHlWZWN0b3JdKTtcbiAgICAgICAgfVxuXG4gICAgICAgIC8vIEhhbmRWZWN0b3Ig44Gu5Zyn57iuXG4gICAgICAgIGxldCBoYW5kVmVjdG9yOiAobnVtYmVyW10gfCBudWxsKVtdIHwgdW5kZWZpbmVkID0gdW5kZWZpbmVkO1xuICAgICAgICBpZiAocG9zZS5oYW5kVmVjdG9ycykge1xuICAgICAgICAgIGhhbmRWZWN0b3IgPSBbXTtcbiAgICAgICAgICBmb3IgKGNvbnN0IGtleSBvZiBQb3NlU2V0LkhBTkRfVkVDVE9SX01BUFBJTkdTKSB7XG4gICAgICAgICAgICBoYW5kVmVjdG9yLnB1c2gocG9zZS5oYW5kVmVjdG9yc1trZXkgYXMga2V5b2YgSGFuZFZlY3Rvcl0pO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuXG4gICAgICAgIC8vIFBvc2VTZXRKc29uSXRlbSDjga4gcG9zZSDjgqrjg5bjgrjjgqfjgq/jg4jjgpLnlJ/miJBcbiAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICB0OiBwb3NlLnRpbWVNaWxpc2Vjb25kcyxcbiAgICAgICAgICBkOiBwb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMsXG4gICAgICAgICAgcDogcG9zZS5wb3NlLFxuICAgICAgICAgIGw6IHBvc2UubGVmdEhhbmQsXG4gICAgICAgICAgcjogcG9zZS5yaWdodEhhbmQsXG4gICAgICAgICAgdjogYm9keVZlY3RvcixcbiAgICAgICAgICBoOiBoYW5kVmVjdG9yLFxuICAgICAgICAgIGU6IHBvc2UuZXh0ZW5kZWREYXRhLFxuICAgICAgICB9O1xuICAgICAgfSksXG4gICAgICBwb3NlTGFuZG1hcmtNYXBwcGluZ3M6IHBvc2VMYW5kbWFya01hcHBpbmdzLFxuICAgIH07XG5cbiAgICByZXR1cm4gSlNPTi5zdHJpbmdpZnkoanNvbik7XG4gIH1cblxuICBsb2FkSnNvbihqc29uOiBzdHJpbmcgfCBhbnkpIHtcbiAgICBjb25zdCBwYXJzZWRKc29uID0gdHlwZW9mIGpzb24gPT09ICdzdHJpbmcnID8gSlNPTi5wYXJzZShqc29uKSA6IGpzb247XG5cbiAgICBpZiAocGFyc2VkSnNvbi5nZW5lcmF0b3IgIT09ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicpIHtcbiAgICAgIHRocm93ICfkuI3mraPjgarjg5XjgqHjgqTjg6snO1xuICAgIH0gZWxzZSBpZiAocGFyc2VkSnNvbi52ZXJzaW9uICE9PSAxKSB7XG4gICAgICB0aHJvdyAn5pyq5a++5b+c44Gu44OQ44O844K444On44OzJztcbiAgICB9XG5cbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSBwYXJzZWRKc29uLnZpZGVvO1xuICAgIHRoaXMucG9zZXMgPSBwYXJzZWRKc29uLnBvc2VzLm1hcCgoaXRlbTogUG9zZVNldEpzb25JdGVtKTogUG9zZVNldEl0ZW0gPT4ge1xuICAgICAgY29uc3QgYm9keVZlY3RvcjogYW55ID0ge307XG4gICAgICBQb3NlU2V0LkJPRFlfVkVDVE9SX01BUFBJTkdTLm1hcCgoa2V5LCBpbmRleCkgPT4ge1xuICAgICAgICBib2R5VmVjdG9yW2tleSBhcyBrZXlvZiBCb2R5VmVjdG9yXSA9IGl0ZW0udltpbmRleF07XG4gICAgICB9KTtcblxuICAgICAgY29uc3QgaGFuZFZlY3RvcjogYW55ID0ge307XG4gICAgICBpZiAoaXRlbS5oKSB7XG4gICAgICAgIFBvc2VTZXQuSEFORF9WRUNUT1JfTUFQUElOR1MubWFwKChrZXksIGluZGV4KSA9PiB7XG4gICAgICAgICAgaGFuZFZlY3RvcltrZXkgYXMga2V5b2YgSGFuZFZlY3Rvcl0gPSBpdGVtLmghW2luZGV4XTtcbiAgICAgICAgfSk7XG4gICAgICB9XG5cbiAgICAgIHJldHVybiB7XG4gICAgICAgIHRpbWVNaWxpc2Vjb25kczogaXRlbS50LFxuICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBpdGVtLmQsXG4gICAgICAgIHBvc2U6IGl0ZW0ucCxcbiAgICAgICAgbGVmdEhhbmQ6IGl0ZW0ubCxcbiAgICAgICAgcmlnaHRIYW5kOiBpdGVtLnIsXG4gICAgICAgIGJvZHlWZWN0b3JzOiBib2R5VmVjdG9yLFxuICAgICAgICBoYW5kVmVjdG9yczogaGFuZFZlY3RvcixcbiAgICAgICAgZnJhbWVJbWFnZURhdGFVcmw6IHVuZGVmaW5lZCxcbiAgICAgICAgZXh0ZW5kZWREYXRhOiBpdGVtLmUsXG4gICAgICB9O1xuICAgIH0pO1xuICB9XG5cbiAgYXN5bmMgbG9hZFppcChidWZmZXI6IEFycmF5QnVmZmVyLCBpbmNsdWRlSW1hZ2VzOiBib29sZWFuID0gdHJ1ZSkge1xuICAgIGNvbnNvbGUubG9nKGBbUG9zZVNldF0gbG9hZFppcC4uLmAsIEpTWmlwKTtcbiAgICBjb25zdCBqc1ppcCA9IG5ldyBKU1ppcCgpO1xuICAgIGNvbnNvbGUubG9nKGBbUG9zZVNldF0gaW5pdC4uLmApO1xuICAgIGNvbnN0IHppcCA9IGF3YWl0IGpzWmlwLmxvYWRBc3luYyhidWZmZXIsIHsgYmFzZTY0OiBmYWxzZSB9KTtcbiAgICBpZiAoIXppcCkgdGhyb3cgJ1pJUOODleOCoeOCpOODq+OCkuiqreOBv+i+vOOCgeOBvuOBm+OCk+OBp+OBl+OBnyc7XG5cbiAgICBjb25zdCBqc29uID0gYXdhaXQgemlwLmZpbGUoJ3Bvc2VzLmpzb24nKT8uYXN5bmMoJ3RleHQnKTtcbiAgICBpZiAoanNvbiA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICB0aHJvdyAnWklQ44OV44Kh44Kk44Or44GrIHBvc2UuanNvbiDjgYzlkKvjgb7jgozjgabjgYTjgb7jgZvjgpMnO1xuICAgIH1cblxuICAgIHRoaXMubG9hZEpzb24oanNvbik7XG5cbiAgICBjb25zdCBmaWxlRXh0ID0gdGhpcy5nZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKHRoaXMuSU1BR0VfTUlNRSk7XG5cbiAgICBpZiAoaW5jbHVkZUltYWdlcykge1xuICAgICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgICAgY29uc3QgZnJhbWVJbWFnZUZpbGVOYW1lID0gYGZyYW1lLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7ZmlsZUV4dH1gO1xuICAgICAgICAgIGNvbnN0IGltYWdlQmFzZTY0ID0gYXdhaXQgemlwXG4gICAgICAgICAgICAuZmlsZShmcmFtZUltYWdlRmlsZU5hbWUpXG4gICAgICAgICAgICA/LmFzeW5jKCdiYXNlNjQnKTtcbiAgICAgICAgICBpZiAoaW1hZ2VCYXNlNjQpIHtcbiAgICAgICAgICAgIHBvc2UuZnJhbWVJbWFnZURhdGFVcmwgPSBgZGF0YToke3RoaXMuSU1BR0VfTUlNRX07YmFzZTY0LCR7aW1hZ2VCYXNlNjR9YDtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgaWYgKCFwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgICBjb25zdCBwb3NlSW1hZ2VGaWxlTmFtZSA9IGBwb3NlLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7ZmlsZUV4dH1gO1xuICAgICAgICAgIGNvbnN0IGltYWdlQmFzZTY0ID0gYXdhaXQgemlwXG4gICAgICAgICAgICAuZmlsZShwb3NlSW1hZ2VGaWxlTmFtZSlcbiAgICAgICAgICAgID8uYXN5bmMoJ2Jhc2U2NCcpO1xuICAgICAgICAgIGlmIChpbWFnZUJhc2U2NCkge1xuICAgICAgICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsID0gYGRhdGE6JHt0aGlzLklNQUdFX01JTUV9O2Jhc2U2NCwke2ltYWdlQmFzZTY0fWA7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG59XG4iXX0=