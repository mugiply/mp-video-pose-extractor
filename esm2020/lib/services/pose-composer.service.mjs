import { Injectable } from '@angular/core';
import { Pose } from '../classes/pose';
import * as i0 from "@angular/core";
/**
 * ポーズを管理するためのサービス
 */
export class PoseComposerService {
    constructor() { }
    init(videoName) {
        const pose = new Pose();
        pose.setVideoName(videoName);
        return pose;
    }
    async downloadAsJson(pose) {
        const blob = new Blob([await pose.getJson()], {
            type: 'application/json',
        });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.target = '_blank';
        a.download = `${pose.getVideoName()}-poses.json`;
        a.click();
    }
    async downloadAsZip(pose) {
        const content = await pose.getZip();
        const url = window.URL.createObjectURL(content);
        const a = document.createElement('a');
        a.href = url;
        a.target = '_blank';
        a.download = `${pose.getVideoName()}-poses.zip`;
        a.click();
    }
}
PoseComposerService.ɵfac = i0.ɵɵngDeclareFactory({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseComposerService, deps: [], target: i0.ɵɵFactoryTarget.Injectable });
PoseComposerService.ɵprov = i0.ɵɵngDeclareInjectable({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseComposerService, providedIn: 'root' });
i0.ɵɵngDeclareClassMetadata({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseComposerService, decorators: [{
            type: Injectable,
            args: [{
                    providedIn: 'root',
                }]
        }], ctorParameters: function () { return []; } });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1jb21wb3Nlci5zZXJ2aWNlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vcHJvamVjdHMvbmd4LW1wLXBvc2UtZXh0cmFjdG9yL3NyYy9saWIvc2VydmljZXMvcG9zZS1jb21wb3Nlci5zZXJ2aWNlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxVQUFVLEVBQUUsTUFBTSxlQUFlLENBQUM7QUFDM0MsT0FBTyxFQUFFLElBQUksRUFBRSxNQUFNLGlCQUFpQixDQUFDOztBQUV2Qzs7R0FFRztBQUlILE1BQU0sT0FBTyxtQkFBbUI7SUFDOUIsZ0JBQWUsQ0FBQztJQUVoQixJQUFJLENBQUMsU0FBaUI7UUFDcEIsTUFBTSxJQUFJLEdBQUcsSUFBSSxJQUFJLEVBQUUsQ0FBQztRQUN4QixJQUFJLENBQUMsWUFBWSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQzdCLE9BQU8sSUFBSSxDQUFDO0lBQ2QsQ0FBQztJQUVELEtBQUssQ0FBQyxjQUFjLENBQUMsSUFBVTtRQUM3QixNQUFNLElBQUksR0FBRyxJQUFJLElBQUksQ0FBQyxDQUFDLE1BQU0sSUFBSSxDQUFDLE9BQU8sRUFBRSxDQUFDLEVBQUU7WUFDNUMsSUFBSSxFQUFFLGtCQUFrQjtTQUN6QixDQUFDLENBQUM7UUFDSCxNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUMsR0FBRyxDQUFDLGVBQWUsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUM3QyxNQUFNLENBQUMsR0FBRyxRQUFRLENBQUMsYUFBYSxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBQ3RDLENBQUMsQ0FBQyxJQUFJLEdBQUcsR0FBRyxDQUFDO1FBQ2IsQ0FBQyxDQUFDLE1BQU0sR0FBRyxRQUFRLENBQUM7UUFDcEIsQ0FBQyxDQUFDLFFBQVEsR0FBRyxHQUFHLElBQUksQ0FBQyxZQUFZLEVBQUUsYUFBYSxDQUFDO1FBQ2pELENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQztJQUNaLENBQUM7SUFFRCxLQUFLLENBQUMsYUFBYSxDQUFDLElBQVU7UUFDNUIsTUFBTSxPQUFPLEdBQUcsTUFBTSxJQUFJLENBQUMsTUFBTSxFQUFFLENBQUM7UUFDcEMsTUFBTSxHQUFHLEdBQUcsTUFBTSxDQUFDLEdBQUcsQ0FBQyxlQUFlLENBQUMsT0FBTyxDQUFDLENBQUM7UUFDaEQsTUFBTSxDQUFDLEdBQUcsUUFBUSxDQUFDLGFBQWEsQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUN0QyxDQUFDLENBQUMsSUFBSSxHQUFHLEdBQUcsQ0FBQztRQUNiLENBQUMsQ0FBQyxNQUFNLEdBQUcsUUFBUSxDQUFDO1FBQ3BCLENBQUMsQ0FBQyxRQUFRLEdBQUcsR0FBRyxJQUFJLENBQUMsWUFBWSxFQUFFLFlBQVksQ0FBQztRQUNoRCxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUM7SUFDWixDQUFDOztnSEE3QlUsbUJBQW1CO29IQUFuQixtQkFBbUIsY0FGbEIsTUFBTTsyRkFFUCxtQkFBbUI7a0JBSC9CLFVBQVU7bUJBQUM7b0JBQ1YsVUFBVSxFQUFFLE1BQU07aUJBQ25CIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgSW5qZWN0YWJsZSB9IGZyb20gJ0Bhbmd1bGFyL2NvcmUnO1xuaW1wb3J0IHsgUG9zZSB9IGZyb20gJy4uL2NsYXNzZXMvcG9zZSc7XG5cbi8qKlxuICog44Od44O844K644KS566h55CG44GZ44KL44Gf44KB44Gu44K144O844OT44K5XG4gKi9cbkBJbmplY3RhYmxlKHtcbiAgcHJvdmlkZWRJbjogJ3Jvb3QnLFxufSlcbmV4cG9ydCBjbGFzcyBQb3NlQ29tcG9zZXJTZXJ2aWNlIHtcbiAgY29uc3RydWN0b3IoKSB7fVxuXG4gIGluaXQodmlkZW9OYW1lOiBzdHJpbmcpOiBQb3NlIHtcbiAgICBjb25zdCBwb3NlID0gbmV3IFBvc2UoKTtcbiAgICBwb3NlLnNldFZpZGVvTmFtZSh2aWRlb05hbWUpO1xuICAgIHJldHVybiBwb3NlO1xuICB9XG5cbiAgYXN5bmMgZG93bmxvYWRBc0pzb24ocG9zZTogUG9zZSkge1xuICAgIGNvbnN0IGJsb2IgPSBuZXcgQmxvYihbYXdhaXQgcG9zZS5nZXRKc29uKCldLCB7XG4gICAgICB0eXBlOiAnYXBwbGljYXRpb24vanNvbicsXG4gICAgfSk7XG4gICAgY29uc3QgdXJsID0gd2luZG93LlVSTC5jcmVhdGVPYmplY3RVUkwoYmxvYik7XG4gICAgY29uc3QgYSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2EnKTtcbiAgICBhLmhyZWYgPSB1cmw7XG4gICAgYS50YXJnZXQgPSAnX2JsYW5rJztcbiAgICBhLmRvd25sb2FkID0gYCR7cG9zZS5nZXRWaWRlb05hbWUoKX0tcG9zZXMuanNvbmA7XG4gICAgYS5jbGljaygpO1xuICB9XG5cbiAgYXN5bmMgZG93bmxvYWRBc1ppcChwb3NlOiBQb3NlKSB7XG4gICAgY29uc3QgY29udGVudCA9IGF3YWl0IHBvc2UuZ2V0WmlwKCk7XG4gICAgY29uc3QgdXJsID0gd2luZG93LlVSTC5jcmVhdGVPYmplY3RVUkwoY29udGVudCk7XG4gICAgY29uc3QgYSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2EnKTtcbiAgICBhLmhyZWYgPSB1cmw7XG4gICAgYS50YXJnZXQgPSAnX2JsYW5rJztcbiAgICBhLmRvd25sb2FkID0gYCR7cG9zZS5nZXRWaWRlb05hbWUoKX0tcG9zZXMuemlwYDtcbiAgICBhLmNsaWNrKCk7XG4gIH1cbn1cbiJdfQ==