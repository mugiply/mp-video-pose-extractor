import { Pose } from '../classes/pose';
import * as i0 from "@angular/core";
/**
 * ポーズを管理するためのサービス
 */
export declare class PoseComposerService {
    constructor();
    init(videoName: string): Pose;
    downloadAsJson(pose: Pose): void;
    downloadAsZip(pose: Pose): Promise<void>;
    static ɵfac: i0.ɵɵFactoryDeclaration<PoseComposerService, never>;
    static ɵprov: i0.ɵɵInjectableDeclaration<PoseComposerService>;
}
