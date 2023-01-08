import { POSE_LANDMARKS } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
// @ts-ignore
import cosSimilarity from 'cos-similarity';
export class Pose {
    constructor() {
        this.poses = [];
        this.isFinalized = false;
        this.videoMetadata = {
            name: '',
            width: 0,
            height: 0,
            duration: 0,
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
    pushPose(videoTimeMiliseconds, frameImageJpegDataUrl, videoWidth, videoHeight, videoDuration, results) {
        this.setVideoMetaData(videoWidth, videoHeight, videoDuration);
        if (results.poseLandmarks === undefined)
            return;
        const poseLandmarksWithWorldCoordinate = results.ea
            ? results.ea
            : [];
        if (poseLandmarksWithWorldCoordinate.length === 0) {
            console.warn(`[Pose] pushPose - Could not get the pose with the world coordinate`, results);
            return;
        }
        const poseVector = Pose.getPoseVector(poseLandmarksWithWorldCoordinate);
        if (!poseVector) {
            console.warn(`[Pose] pushPose - Could not get the pose vector`, poseLandmarksWithWorldCoordinate);
            return;
        }
        const pose = {
            t: videoTimeMiliseconds,
            pose: poseLandmarksWithWorldCoordinate.map((landmark) => {
                return [landmark.x, landmark.y, landmark.z, landmark.visibility];
            }),
            vectors: poseVector,
            frameImageDataUrl: frameImageJpegDataUrl,
        };
        if (1 <= this.poses.length) {
            const lastPose = this.poses[this.poses.length - 1];
            if (Pose.isSimilarPose(lastPose.vectors, pose.vectors)) {
                return;
            }
        }
        this.poses.push(pose);
    }
    finalize() {
        if (Pose.IS_ENABLE_DUPLICATED_POSE_REDUCTION) {
            // 全ポーズを走査して、類似するポーズを削除する
            const newPoses = [];
            for (const poseA of this.poses) {
                let isDuplicated = false;
                for (const poseB of newPoses) {
                    if (Pose.isSimilarPose(poseA.vectors, poseB.vectors)) {
                        isDuplicated = true;
                        break;
                    }
                }
                if (isDuplicated)
                    continue;
                newPoses.push(poseA);
            }
            console.info(`[Pose] getJson - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`);
            this.poses = newPoses;
        }
        this.isFinalized = true;
    }
    getSimilarPoses(results) {
        const poseVector = Pose.getPoseVector(results.ea);
        if (!poseVector)
            throw 'Could not get the pose vector';
        return this.poses.filter((p) => Pose.isSimilarPose(p.vectors, poseVector));
    }
    static getPoseVector(poseLandmarks) {
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
    static isSimilarPose(poseVectorA, poseVectorB, threshold = 0.9) {
        const cosSimilarities = {
            leftWristToLeftElbow: cosSimilarity(poseVectorA.leftWristToLeftElbow, poseVectorB.leftWristToLeftElbow),
            leftElbowToLeftShoulder: cosSimilarity(poseVectorA.leftElbowToLeftShoulder, poseVectorB.leftElbowToLeftShoulder),
            rightWristToRightElbow: cosSimilarity(poseVectorA.rightWristToRightElbow, poseVectorB.rightWristToRightElbow),
            rightElbowToRightShoulder: cosSimilarity(poseVectorA.rightElbowToRightShoulder, poseVectorB.rightElbowToRightShoulder),
        };
        let isSimilar = false;
        const cosSimilaritiesSum = Object.values(cosSimilarities).reduce((sum, value) => sum + value, 0);
        if (cosSimilaritiesSum >= threshold * Object.keys(cosSimilarities).length)
            isSimilar = true;
        console.log(`[Pose] isSimilarPose`, isSimilar, cosSimilarities);
        return isSimilar;
    }
    async getZip() {
        const jsZip = new JSZip();
        jsZip.file('poses.json', this.getJson());
        for (const pose of this.poses) {
            if (!pose.frameImageDataUrl)
                continue;
            try {
                const index = pose.frameImageDataUrl.indexOf('base64,') + 'base64,'.length;
                const base64 = pose.frameImageDataUrl.substring(index);
                jsZip.file(`snapshot-${pose.t}.jpg`, base64, { base64: true });
            }
            catch (error) {
                console.warn(`[PoseExporterService] push - Could not push frame image`, error);
                throw error;
            }
        }
        return await jsZip.generateAsync({ type: 'blob' });
    }
    getJson() {
        if (this.videoMetadata === undefined || this.poses === undefined)
            return '{}';
        if (!this.isFinalized) {
            this.finalize();
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
                const poseVector = [];
                for (const key of Pose.POSE_VECTOR_MAPPINGS) {
                    poseVector.push(pose.vectors[key]);
                }
                return {
                    t: pose.t,
                    pose: Pose.IS_SHRINK_RAW_POSE_DATA ? [] : pose.pose,
                    vectors: poseVector,
                };
            }),
            poseLandmarkMapppings: Pose.IS_SHRINK_RAW_POSE_DATA
                ? []
                : poseLandmarkMappings,
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
        this.poses = parsedJson.poses.map((poseJsonItem) => {
            const poseVector = {};
            Pose.POSE_VECTOR_MAPPINGS.map((key, index) => {
                poseVector[key] = poseJsonItem.vectors[index];
            });
            return {
                t: poseJsonItem.t,
                pose: poseJsonItem.pose,
                vectors: poseVector,
                frameImageDataUrl: undefined,
            };
        });
    }
    async loadZip(buffer, includeImages = true) {
        const jsZip = new JSZip();
        const zip = await jsZip.loadAsync(buffer, { base64: false });
        if (!zip)
            throw 'ZIPファイルを読み込めませんでした';
        const json = await zip.file('poses.json')?.async('text');
        if (json === undefined) {
            throw 'ZIPファイルに pose.json が含まれていません';
        }
        this.loadJson(json);
        if (includeImages) {
            for (const pose of this.poses) {
                const frameImageFileName = `snapshot-${pose.t}.jpg`;
                const imageBase64 = await zip.file(frameImageFileName)?.async('base64');
                if (imageBase64 === undefined && !pose.frameImageDataUrl) {
                    continue;
                }
                pose.frameImageDataUrl = `data:image/jpeg;base64,${imageBase64}`;
            }
        }
    }
}
Pose.IS_ENABLE_DUPLICATED_POSE_REDUCTION = true;
Pose.IS_SHRINK_RAW_POSE_DATA = false;
Pose.POSE_VECTOR_MAPPINGS = [
    'rightWristToRightElbow',
    'rightElbowToRightShoulder',
    'leftWristToLeftElbow',
    'leftElbowToLeftShoulder',
];
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3Byb2plY3RzL25neC1tcC1wb3NlLWV4dHJhY3Rvci9zcmMvbGliL2NsYXNzZXMvcG9zZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSxPQUFPLEVBQUUsY0FBYyxFQUFXLE1BQU0scUJBQXFCLENBQUM7QUFDOUQsT0FBTyxLQUFLLEtBQUssTUFBTSxPQUFPLENBQUM7QUFNL0IsYUFBYTtBQUNiLE9BQU8sYUFBYSxNQUFNLGdCQUFnQixDQUFDO0FBRTNDLE1BQU0sT0FBTyxJQUFJO0lBdUJmO1FBZE8sVUFBSyxHQUFlLEVBQUUsQ0FBQztRQUN2QixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQWNuQyxJQUFJLENBQUMsYUFBYSxHQUFHO1lBQ25CLElBQUksRUFBRSxFQUFFO1lBQ1IsS0FBSyxFQUFFLENBQUM7WUFDUixNQUFNLEVBQUUsQ0FBQztZQUNULFFBQVEsRUFBRSxDQUFDO1NBQ1osQ0FBQztJQUNKLENBQUM7SUFFRCxZQUFZO1FBQ1YsT0FBTyxJQUFJLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQztJQUNqQyxDQUFDO0lBRUQsWUFBWSxDQUFDLFNBQWlCO1FBQzVCLElBQUksQ0FBQyxhQUFhLENBQUMsSUFBSSxHQUFHLFNBQVMsQ0FBQztJQUN0QyxDQUFDO0lBRUQsZ0JBQWdCLENBQUMsS0FBYSxFQUFFLE1BQWMsRUFBRSxRQUFnQjtRQUM5RCxJQUFJLENBQUMsYUFBYSxDQUFDLEtBQUssR0FBRyxLQUFLLENBQUM7UUFDakMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxNQUFNLEdBQUcsTUFBTSxDQUFDO1FBQ25DLElBQUksQ0FBQyxhQUFhLENBQUMsUUFBUSxHQUFHLFFBQVEsQ0FBQztJQUN6QyxDQUFDO0lBRUQsZ0JBQWdCO1FBQ2QsSUFBSSxJQUFJLENBQUMsS0FBSyxLQUFLLFNBQVM7WUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO1FBQ3hDLE9BQU8sSUFBSSxD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VyxLQUFLLFNBQVMsSUFBSSxDQUFDLElBQUksQ0FBQyxpQkFBaUIsRUFBRTtvQkFDeEQsU0FBUztpQkFDVjtnQkFFRCxJQUFJLENBQUMsaUJBQWlCLEdBQUcsMEJBQTBCLFdBQVcsRUFBRSxDQUFDO2FBQ2xFO1NBQ0Y7SUFDSCxDQUFDOztBQTVUc0Isd0NBQW1DLEdBQUcsSUFBSSxDQUFDO0FBRTNDLDRCQUF1QixHQUFHLEtBQUssQ0FBQztBQUVoQyx5QkFBb0IsR0FBRztJQUM1Qyx3QkFBd0I7SUFDeEIsMkJBQTJCO0lBQzNCLHNCQUFzQjtJQUN0Qix5QkFBeUI7Q0FDMUIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IFBPU0VfTEFORE1BUktTLCBSZXN1bHRzIH0gZnJvbSAnQG1lZGlhcGlwZS9ob2xpc3RpYyc7XG5pbXBvcnQgKiBhcyBKU1ppcCBmcm9tICdqc3ppcCc7XG5pbXBvcnQgeyBQb3NlSXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1pdGVtJztcbmltcG9ydCB7IFBvc2VKc29uIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLWpzb24nO1xuaW1wb3J0IHsgUG9zZUpzb25JdGVtIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLWpzb24taXRlbSc7XG5pbXBvcnQgeyBQb3NlVmVjdG9yIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXZlY3Rvcic7XG5cbi8vIEB0cy1pZ25vcmVcbmltcG9ydCBjb3NTaW1pbGFyaXR5IGZyb20gJ2Nvcy1zaW1pbGFyaXR5JztcblxuZXhwb3J0IGNsYXNzIFBvc2Uge1xuICBwdWJsaWMgZ2VuZXJhdG9yPzogc3RyaW5nO1xuICBwdWJsaWMgdmVyc2lvbj86IG51bWJlcjtcbiAgcHJpdmF0ZSB2aWRlb01ldGFkYXRhIToge1xuICAgIG5hbWU6IHN0cmluZztcbiAgICB3aWR0aDogbnVtYmVyO1xuICAgIGhlaWdodDogbnVtYmVyO1xuICAgIGR1cmF0aW9uOiBudW1iZXI7XG4gIH07XG4gIHB1YmxpYyBwb3NlczogUG9zZUl0ZW1bXSA9IFtdO1xuICBwdWJsaWMgaXNGaW5hbGl6ZWQ/OiBib29sZWFuID0gZmFsc2U7XG5cbiAgcHVibGljIHN0YXRpYyByZWFkb25seSBJU19FTkFCTEVfRFVQTElDQVRFRF9QT1NFX1JFRFVDVElPTiA9IHRydWU7XG5cbiAgcHVibGljIHN0YXRpYyByZWFkb25seSBJU19TSFJJTktfUkFXX1BPU0VfREFUQSA9IGZhbHNlO1xuXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgUE9TRV9WRUNUT1JfTUFQUElOR1MgPSBbXG4gICAgJ3JpZ2h0V3Jpc3RUb1JpZ2h0RWxib3cnLFxuICAgICdyaWdodEVsYm93VG9SaWdodFNob3VsZGVyJyxcbiAgICAnbGVmdFdyaXN0VG9MZWZ0RWxib3cnLFxuICAgICdsZWZ0RWxib3dUb0xlZnRTaG91bGRlcicsXG4gIF07XG5cbiAgY29uc3RydWN0b3IoKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhID0ge1xuICAgICAgbmFtZTogJycsXG4gICAgICB3aWR0aDogMCxcbiAgICAgIGhlaWdodDogMCxcbiAgICAgIGR1cmF0aW9uOiAwLFxuICAgIH07XG4gIH1cblxuICBnZXRWaWRlb05hbWUoKSB7XG4gICAgcmV0dXJuIHRoaXMudmlkZW9NZXRhZGF0YS5uYW1lO1xuICB9XG5cbiAgc2V0VmlkZW9OYW1lKHZpZGVvTmFtZTogc3RyaW5nKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLm5hbWUgPSB2aWRlb05hbWU7XG4gIH1cblxuICBzZXRWaWRlb01ldGFEYXRhKHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLCBkdXJhdGlvbjogbnVtYmVyKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLndpZHRoID0gd2lkdGg7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLmhlaWdodCA9IGhlaWdodDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24gPSBkdXJhdGlvbjtcbiAgfVxuXG4gIGdldE51bWJlck9mUG9zZXMoKTogbnVtYmVyIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gLTE7XG4gICAgcmV0dXJuIHRoaXMucG9zZXMubGVuZ3RoO1xuICB9XG5cbiAgZ2V0UG9zZXMoKTogUG9zZUl0ZW1bXSB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIFtdO1xuICAgIHJldHVybiB0aGlzLnBvc2VzO1xuICB9XG5cbiAgcHVzaFBvc2UoXG4gICAgdmlkZW9UaW1lTWlsaXNlY29uZHM6IG51bWJlcixcbiAgICBmcmFtZUltYWdlSnBlZ0RhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICB2aWRlb1dpZHRoOiBudW1iZXIsXG4gICAgdmlkZW9IZWlnaHQ6IG51bWJlcixcbiAgICB2aWRlb0R1cmF0aW9uOiBudW1iZXIsXG4gICAgcmVzdWx0czogUmVzdWx0c1xuICApIHtcbiAgICB0aGlzLnNldFZpZGVvTWV0YURhdGEodmlkZW9XaWR0aCwgdmlkZW9IZWlnaHQsIHZpZGVvRHVyYXRpb24pO1xuXG4gICAgaWYgKHJlc3VsdHMucG9zZUxhbmRtYXJrcyA9PT0gdW5kZWZpbmVkKSByZXR1cm47XG5cbiAgICBjb25zdCBwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZTogYW55W10gPSAocmVzdWx0cyBhcyBhbnkpLmVhXG4gICAgICA/IChyZXN1bHRzIGFzIGFueSkuZWFcbiAgICAgIDogW107XG4gICAgaWYgKHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlLmxlbmd0aCA9PT0gMCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VdIHB1c2hQb3NlIC0gQ291bGQgbm90IGdldCB0aGUgcG9zZSB3aXRoIHRoZSB3b3JsZCBjb29yZGluYXRlYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICBjb25zdCBwb3NlVmVjdG9yID0gUG9zZS5nZXRQb3NlVmVjdG9yKHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlKTtcbiAgICBpZiAoIXBvc2VWZWN0b3IpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlXSBwdXNoUG9zZSAtIENvdWxkIG5vdCBnZXQgdGhlIHBvc2UgdmVjdG9yYCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGVcbiAgICAgICk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3QgcG9zZTogUG9zZUl0ZW0gPSB7XG4gICAgICB0OiB2aWRlb1RpbWVNaWxpc2Vjb25kcyxcbiAgICAgIHBvc2U6IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlLm1hcCgobGFuZG1hcmspID0+IHtcbiAgICAgICAgcmV0dXJuIFtsYW5kbWFyay54LCBsYW5kbWFyay55LCBsYW5kbWFyay56LCBsYW5kbWFyay52aXNpYmlsaXR5XTtcbiAgICAgIH0pLFxuICAgICAgdmVjdG9yczogcG9zZVZlY3RvcixcbiAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiBmcmFtZUltYWdlSnBlZ0RhdGFVcmwsXG4gICAgfTtcblxuICAgIGlmICgxIDw9IHRoaXMucG9zZXMubGVuZ3RoKSB7XG4gICAgICBjb25zdCBsYXN0UG9zZSA9IHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXTtcbiAgICAgIGlmIChQb3NlLmlzU2ltaWxhclBvc2UobGFzdFBvc2UudmVjdG9ycywgcG9zZS52ZWN0b3JzKSkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgfVxuXG4gICAgdGhpcy5wb3Nlcy5wdXNoKHBvc2UpO1xuICB9XG5cbiAgZmluYWxpemUoKSB7XG4gICAgaWYgKFBvc2UuSVNfRU5BQkxFX0RVUExJQ0FURURfUE9TRV9SRURVQ1RJT04pIHtcbiAgICAgIC8vIOWFqOODneODvOOCuuOCkui1sOafu+OBl+OBpuOAgemhnuS8vOOBmeOCi+ODneODvOOCuuOCkuWJiumZpOOBmeOCi1xuICAgICAgY29uc3QgbmV3UG9zZXM6IFBvc2VJdGVtW10gPSBbXTtcbiAgICAgIGZvciAoY29uc3QgcG9zZUEgb2YgdGhpcy5wb3Nlcykge1xuICAgICAgICBsZXQgaXNEdXBsaWNhdGVkID0gZmFsc2U7XG4gICAgICAgIGZvciAoY29uc3QgcG9zZUIgb2YgbmV3UG9zZXMpIHtcbiAgICAgICAgICBpZiAoUG9zZS5pc1NpbWlsYXJQb3NlKHBvc2VBLnZlY3RvcnMsIHBvc2VCLnZlY3RvcnMpKSB7XG4gICAgICAgICAgICBpc0R1cGxpY2F0ZWQgPSB0cnVlO1xuICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIGlmIChpc0R1cGxpY2F0ZWQpIGNvbnRpbnVlO1xuXG4gICAgICAgIG5ld1Bvc2VzLnB1c2gocG9zZUEpO1xuICAgICAgfVxuXG4gICAgICBjb25zb2xlLmluZm8oXG4gICAgICAgIGBbUG9zZV0gZ2V0SnNvbiAtIFJlZHVjZWQgJHt0aGlzLnBvc2VzLmxlbmd0aH0gcG9zZXMgLT4gJHtuZXdQb3Nlcy5sZW5ndGh9IHBvc2VzYFxuICAgICAgKTtcbiAgICAgIHRoaXMucG9zZXMgPSBuZXdQb3NlcztcbiAgICB9XG5cbiAgICB0aGlzLmlzRmluYWxpemVkID0gdHJ1ZTtcbiAgfVxuXG4gIGdldFNpbWlsYXJQb3NlcyhyZXN1bHRzOiBSZXN1bHRzKTogUG9zZUl0ZW1bXSB7XG4gICAgY29uc3QgcG9zZVZlY3RvciA9IFBvc2UuZ2V0UG9zZVZlY3RvcigocmVzdWx0cyBhcyBhbnkpLmVhKTtcbiAgICBpZiAoIXBvc2VWZWN0b3IpIHRocm93ICdDb3VsZCBub3QgZ2V0IHRoZSBwb3NlIHZlY3Rvcic7XG5cbiAgICByZXR1cm4gdGhpcy5wb3Nlcy5maWx0ZXIoKHApID0+IFBvc2UuaXNTaW1pbGFyUG9zZShwLnZlY3RvcnMsIHBvc2VWZWN0b3IpKTtcbiAgfVxuXG4gIHN0YXRpYyBnZXRQb3NlVmVjdG9yKFxuICAgIHBvc2VMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W11cbiAgKTogUG9zZVZlY3RvciB8IHVuZGVmaW5lZCB7XG4gICAgcmV0dXJuIHtcbiAgICAgIHJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICByaWdodEVsYm93VG9SaWdodFNob3VsZGVyOiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnosXG4gICAgICBdLFxuICAgICAgbGVmdFdyaXN0VG9MZWZ0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS56LFxuICAgICAgXSxcbiAgICB9O1xuICB9XG5cbiAgc3RhdGljIGlzU2ltaWxhclBvc2UoXG4gICAgcG9zZVZlY3RvckE6IFBvc2VWZWN0b3IsXG4gICAgcG9zZVZlY3RvckI6IFBvc2VWZWN0b3IsXG4gICAgdGhyZXNob2xkID0gMC45XG4gICk6IGJvb2xlYW4ge1xuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllcyA9IHtcbiAgICAgIGxlZnRXcmlzdFRvTGVmdEVsYm93OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBwb3NlVmVjdG9yQS5sZWZ0V3Jpc3RUb0xlZnRFbGJvdyxcbiAgICAgICAgcG9zZVZlY3RvckIubGVmdFdyaXN0VG9MZWZ0RWxib3dcbiAgICAgICksXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogY29zU2ltaWxhcml0eShcbiAgICAgICAgcG9zZVZlY3RvckEubGVmdEVsYm93VG9MZWZ0U2hvdWxkZXIsXG4gICAgICAgIHBvc2VWZWN0b3JCLmxlZnRFbGJvd1RvTGVmdFNob3VsZGVyXG4gICAgICApLFxuICAgICAgcmlnaHRXcmlzdFRvUmlnaHRFbGJvdzogY29zU2ltaWxhcml0eShcbiAgICAgICAgcG9zZVZlY3RvckEucmlnaHRXcmlzdFRvUmlnaHRFbGJvdyxcbiAgICAgICAgcG9zZVZlY3RvckIucmlnaHRXcmlzdFRvUmlnaHRFbGJvd1xuICAgICAgKSxcbiAgICAgIHJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXI6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIHBvc2VWZWN0b3JBLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXIsXG4gICAgICAgIHBvc2VWZWN0b3JCLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXJcbiAgICAgICksXG4gICAgfTtcblxuICAgIGxldCBpc1NpbWlsYXIgPSBmYWxzZTtcbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNTdW0gPSBPYmplY3QudmFsdWVzKGNvc1NpbWlsYXJpdGllcykucmVkdWNlKFxuICAgICAgKHN1bSwgdmFsdWUpID0+IHN1bSArIHZhbHVlLFxuICAgICAgMFxuICAgICk7XG4gICAgaWYgKGNvc1NpbWlsYXJpdGllc1N1bSA+PSB0aHJlc2hvbGQgKiBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXMpLmxlbmd0aClcbiAgICAgIGlzU2ltaWxhciA9IHRydWU7XG5cbiAgICBjb25zb2xlLmxvZyhgW1Bvc2VdIGlzU2ltaWxhclBvc2VgLCBpc1NpbWlsYXIsIGNvc1NpbWlsYXJpdGllcyk7XG5cbiAgICByZXR1cm4gaXNTaW1pbGFyO1xuICB9XG5cbiAgcHVibGljIGFzeW5jIGdldFppcCgpOiBQcm9taXNlPEJsb2I+IHtcbiAgICBjb25zdCBqc1ppcCA9IG5ldyBKU1ppcCgpO1xuICAgIGpzWmlwLmZpbGUoJ3Bvc2VzLmpzb24nLCB0aGlzLmdldEpzb24oKSk7XG5cbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSBjb250aW51ZTtcbiAgICAgIHRyeSB7XG4gICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UuZnJhbWVJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcblxuICAgICAgICBqc1ppcC5maWxlKGBzbmFwc2hvdC0ke3Bvc2UudH0uanBnYCwgYmFzZTY0LCB7IGJhc2U2NDogdHJ1ZSB9KTtcbiAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICBgW1Bvc2VFeHBvcnRlclNlcnZpY2VdIHB1c2ggLSBDb3VsZCBub3QgcHVzaCBmcmFtZSBpbWFnZWAsXG4gICAgICAgICAgZXJyb3JcbiAgICAgICAgKTtcbiAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGF3YWl0IGpzWmlwLmdlbmVyYXRlQXN5bmMoeyB0eXBlOiAnYmxvYicgfSk7XG4gIH1cblxuICBwdWJsaWMgZ2V0SnNvbigpOiBzdHJpbmcge1xuICAgIGlmICh0aGlzLnZpZGVvTWV0YWRhdGEgPT09IHVuZGVmaW5lZCB8fCB0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpXG4gICAgICByZXR1cm4gJ3t9JztcblxuICAgIGlmICghdGhpcy5pc0ZpbmFsaXplZCkge1xuICAgICAgdGhpcy5maW5hbGl6ZSgpO1xuICAgIH1cblxuICAgIGxldCBwb3NlTGFuZG1hcmtNYXBwaW5ncyA9IFtdO1xuICAgIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKFBPU0VfTEFORE1BUktTKSkge1xuICAgICAgY29uc3QgaW5kZXg6IG51bWJlciA9IFBPU0VfTEFORE1BUktTW2tleSBhcyBrZXlvZiB0eXBlb2YgUE9TRV9MQU5ETUFSS1NdO1xuICAgICAgcG9zZUxhbmRtYXJrTWFwcGluZ3NbaW5kZXhdID0ga2V5O1xuICAgIH1cblxuICAgIGNvbnN0IGpzb246IFBvc2VKc29uID0ge1xuICAgICAgZ2VuZXJhdG9yOiAnbXAtdmlkZW8tcG9zZS1leHRyYWN0b3InLFxuICAgICAgdmVyc2lvbjogMSxcbiAgICAgIHZpZGVvOiB0aGlzLnZpZGVvTWV0YWRhdGEhLFxuICAgICAgcG9zZXM6IHRoaXMucG9zZXMubWFwKChwb3NlOiBQb3NlSXRlbSk6IFBvc2VKc29uSXRlbSA9PiB7XG4gICAgICAgIGNvbnN0IHBvc2VWZWN0b3IgPSBbXTtcbiAgICAgICAgZm9yIChjb25zdCBrZXkgb2YgUG9zZS5QT1NFX1ZFQ1RPUl9NQVBQSU5HUykge1xuICAgICAgICAgIHBvc2VWZWN0b3IucHVzaChwb3NlLnZlY3RvcnNba2V5IGFzIGtleW9mIFBvc2VWZWN0b3JdKTtcbiAgICAgICAgfVxuXG4gICAgICAgIHJldHVybiB7XG4gICAgICAgICAgdDogcG9zZS50LFxuICAgICAgICAgIHBvc2U6IFBvc2UuSVNfU0hSSU5LX1JBV19QT1NFX0RBVEEgPyBbXSA6IHBvc2UucG9zZSxcbiAgICAgICAgICB2ZWN0b3JzOiBwb3NlVmVjdG9yLFxuICAgICAgICB9O1xuICAgICAgfSksXG4gICAgICBwb3NlTGFuZG1hcmtNYXBwcGluZ3M6IFBvc2UuSVNfU0hSSU5LX1JBV19QT1NFX0RBVEFcbiAgICAgICAgPyBbXVxuICAgICAgICA6IHBvc2VMYW5kbWFya01hcHBpbmdzLFxuICAgIH07XG5cbiAgICByZXR1cm4gSlNPTi5zdHJpbmdpZnkoanNvbik7XG4gIH1cblxuICBsb2FkSnNvbihqc29uOiBzdHJpbmcgfCBhbnkpIHtcbiAgICBjb25zdCBwYXJzZWRKc29uID0gdHlwZW9mIGpzb24gPT09ICdzdHJpbmcnID8gSlNPTi5wYXJzZShqc29uKSA6IGpzb247XG5cbiAgICBpZiAocGFyc2VkSnNvbi5nZW5lcmF0b3IgIT09ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicpIHtcbiAgICAgIHRocm93ICfkuI3mraPjgarjg5XjgqHjgqTjg6snO1xuICAgIH0gZWxzZSBpZiAocGFyc2VkSnNvbi52ZXJzaW9uICE9PSAxKSB7XG4gICAgICB0aHJvdyAn5pyq5a++5b+c44Gu44OQ44O844K444On44OzJztcbiAgICB9XG5cbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSBwYXJzZWRKc29uLnZpZGVvO1xuICAgIHRoaXMucG9zZXMgPSBwYXJzZWRKc29uLnBvc2VzLm1hcChcbiAgICAgIChwb3NlSnNvbkl0ZW06IFBvc2VKc29uSXRlbSk6IFBvc2VJdGVtID0+IHtcbiAgICAgICAgY29uc3QgcG9zZVZlY3RvcjogYW55ID0ge307XG4gICAgICAgIFBvc2UuUE9TRV9WRUNUT1JfTUFQUElOR1MubWFwKChrZXksIGluZGV4KSA9PiB7XG4gICAgICAgICAgcG9zZVZlY3RvcltrZXkgYXMga2V5b2YgUG9zZVZlY3Rvcl0gPSBwb3NlSnNvbkl0ZW0udmVjdG9yc1tpbmRleF07XG4gICAgICAgIH0pO1xuXG4gICAgICAgIHJldHVybiB7XG4gICAgICAgICAgdDogcG9zZUpzb25JdGVtLnQsXG4gICAgICAgICAgcG9zZTogcG9zZUpzb25JdGVtLnBvc2UsXG4gICAgICAgICAgdmVjdG9yczogcG9zZVZlY3RvcixcbiAgICAgICAgICBmcmFtZUltYWdlRGF0YVVybDogdW5kZWZpbmVkLFxuICAgICAgICB9O1xuICAgICAgfVxuICAgICk7XG4gIH1cblxuICBhc3luYyBsb2FkWmlwKGJ1ZmZlcjogQXJyYXlCdWZmZXIsIGluY2x1ZGVJbWFnZXM6IGJvb2xlYW4gPSB0cnVlKSB7XG4gICAgY29uc3QganNaaXAgPSBuZXcgSlNaaXAoKTtcbiAgICBjb25zdCB6aXAgPSBhd2FpdCBqc1ppcC5sb2FkQXN5bmMoYnVmZmVyLCB7IGJhc2U2NDogZmFsc2UgfSk7XG4gICAgaWYgKCF6aXApIHRocm93ICdaSVDjg5XjgqHjgqTjg6vjgpLoqq3jgb/ovrzjgoHjgb7jgZvjgpPjgafjgZfjgZ8nO1xuXG4gICAgY29uc3QganNvbiA9IGF3YWl0IHppcC5maWxlKCdwb3Nlcy5qc29uJyk/LmFzeW5jKCd0ZXh0Jyk7XG4gICAgaWYgKGpzb24gPT09IHVuZGVmaW5lZCkge1xuICAgICAgdGhyb3cgJ1pJUOODleOCoeOCpOODq+OBqyBwb3NlLmpzb24g44GM5ZCr44G+44KM44Gm44GE44G+44Gb44KTJztcbiAgICB9XG5cbiAgICB0aGlzLmxvYWRKc29uKGpzb24pO1xuXG4gICAgaWYgKGluY2x1ZGVJbWFnZXMpIHtcbiAgICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICAgIGNvbnN0IGZyYW1lSW1hZ2VGaWxlTmFtZSA9IGBzbmFwc2hvdC0ke3Bvc2UudH0uanBnYDtcbiAgICAgICAgY29uc3QgaW1hZ2VCYXNlNjQgPSBhd2FpdCB6aXAuZmlsZShmcmFtZUltYWdlRmlsZU5hbWUpPy5hc3luYygnYmFzZTY0Jyk7XG4gICAgICAgIGlmIChpbWFnZUJhc2U2NCA9PT0gdW5kZWZpbmVkICYmICFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cblxuICAgICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsID0gYGRhdGE6aW1hZ2UvanBlZztiYXNlNjQsJHtpbWFnZUJhc2U2NH1gO1xuICAgICAgfVxuICAgIH1cbiAgfVxufVxuIl19