import { PoseSet } from '../classes/pose-set';
import * as i0 from "@angular/core";
/**
 * ポーズを管理するためのサービス
 */
export declare class PoseComposerService {
    constructor();
    init(videoName: string): PoseSet;
    downloadAsJson(poseSet: PoseSet): Promise<void>;
    downloadAsZip(poseSet: PoseSet): Promise<void>;
    static ɵfac: i0.ɵɵFactoryDeclaration<PoseComposerService, never>;
    static ɵprov: i0.ɵɵInjectableDeclaration<PoseComposerService>;
}
