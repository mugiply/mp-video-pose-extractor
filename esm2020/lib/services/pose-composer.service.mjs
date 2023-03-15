import { Injectable } from '@angular/core';
import { PoseSet } from '../classes/pose-set';
import * as i0 from "@angular/core";
/**
 * ポーズを管理するためのサービス
 */
export class PoseComposerService {
    constructor() { }
    init(videoName) {
        const poseSet = new PoseSet();
        poseSet.setVideoName(videoName);
        return poseSet;
    }
    async downloadAsJson(poseSet) {
        const blob = new Blob([await poseSet.getJson()], {
            type: 'application/json',
        });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.target = '_blank';
        a.download = `${poseSet.getVideoName()}.poses.json`;
        a.click();
    }
    async downloadAsZip(poseSet) {
        const content = await poseSet.getZip();
        const url = window.URL.createObjectURL(content);
        const a = document.createElement('a');
        a.href = url;
        a.target = '_blank';
        a.download = `${poseSet.getVideoName()}.poses.zip`;
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1jb21wb3Nlci5zZXJ2aWNlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vcHJvamVjdHMvbmd4LW1wLXBvc2UtZXh0cmFjdG9yL3NyYy9saWIvc2VydmljZXMvcG9zZS1jb21wb3Nlci5zZXJ2aWNlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxVQUFVLEVBQUUsTUFBTSxlQUFlLENBQUM7QUFDM0MsT0FBTyxFQUFFLE9BQU8sRUFBRSxNQUFNLHFCQUFxQixDQUFDOztBQUU5Qzs7R0FFRztBQUlILE1BQU0sT0FBTyxtQkFBbUI7SUFDOUIsZ0JBQWUsQ0FBQztJQUVoQixJQUFJLENBQUMsU0FBaUI7UUFDcEIsTUFBTSxPQUFPLEdBQUcsSUFBSSxPQUFPLEVBQUUsQ0FBQztRQUM5QixPQUFPLENBQUMsWUFBWSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQ2hDLE9BQU8sT0FBTyxDQUFDO0lBQ2pCLENBQUM7SUFFRCxLQUFLLENBQUMsY0FBYyxDQUFDLE9BQWdCO1FBQ25DLE1BQU0sSUFBSSxHQUFHLElBQUksSUFBSSxDQUFDLENBQUMsTUFBTSxPQUFPLENBQUMsT0FBTyxFQUFFLENBQUMsRUFBRTtZQUMvQyxJQUFJLEVBQUUsa0JBQWtCO1NBQ3pCLENBQUMsQ0FBQztRQUNILE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQyxHQUFHLENBQUMsZUFBZSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQzdDLE1BQU0sQ0FBQyxHQUFHLFFBQVEsQ0FBQyxhQUFhLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDdEMsQ0FBQyxDQUFDLElBQUksR0FBRyxHQUFHLENBQUM7UUFDYixDQUFDLENBQUMsTUFBTSxHQUFHLFFBQVEsQ0FBQztRQUNwQixDQUFDLENBQUMsUUFBUSxHQUFHLEdBQUcsT0FBTyxDQUFDLFlBQVksRUFBRSxhQUFhLENBQUM7UUFDcEQsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDO0lBQ1osQ0FBQztJQUVELEtBQUssQ0FBQyxhQUFhLENBQUMsT0FBZ0I7UUFDbEMsTUFBTSxPQUFPLEdBQUcsTUFBTSxPQUFPLENBQUMsTUFBTSxFQUFFLENBQUM7UUFDdkMsTUFBTSxHQUFHLEdBQUcsTUFBTSxDQUFDLEdBQUcsQ0FBQyxlQUFlLENBQUMsT0FBTyxDQUFDLENBQUM7UUFDaEQsTUFBTSxDQUFDLEdBQUcsUUFBUSxDQUFDLGFBQWEsQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUN0QyxDQUFDLENBQUMsSUFBSSxHQUFHLEdBQUcsQ0FBQztRQUNiLENBQUMsQ0FBQyxNQUFNLEdBQUcsUUFBUSxDQUFDO1FBQ3BCLENBQUMsQ0FBQyxRQUFRLEdBQUcsR0FBRyxPQUFPLENBQUMsWUFBWSxFQUFFLFlBQVksQ0FBQztRQUNuRCxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUM7SUFDWixDQUFDOztnSEE3QlUsbUJBQW1CO29IQUFuQixtQkFBbUIsY0FGbEIsTUFBTTsyRkFFUCxtQkFBbUI7a0JBSC9CLFVBQVU7bUJBQUM7b0JBQ1YsVUFBVSxFQUFFLE1BQU07aUJBQ25CIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgSW5qZWN0YWJsZSB9IGZyb20gJ0Bhbmd1bGFyL2NvcmUnO1xuaW1wb3J0IHsgUG9zZVNldCB9IGZyb20gJy4uL2NsYXNzZXMvcG9zZS1zZXQnO1xuXG4vKipcbiAqIOODneODvOOCuuOCkueuoeeQhuOBmeOCi+OBn+OCgeOBruOCteODvOODk+OCuVxuICovXG5ASW5qZWN0YWJsZSh7XG4gIHByb3ZpZGVkSW46ICdyb290Jyxcbn0pXG5leHBvcnQgY2xhc3MgUG9zZUNvbXBvc2VyU2VydmljZSB7XG4gIGNvbnN0cnVjdG9yKCkge31cblxuICBpbml0KHZpZGVvTmFtZTogc3RyaW5nKTogUG9zZVNldCB7XG4gICAgY29uc3QgcG9zZVNldCA9IG5ldyBQb3NlU2V0KCk7XG4gICAgcG9zZVNldC5zZXRWaWRlb05hbWUodmlkZW9OYW1lKTtcbiAgICByZXR1cm4gcG9zZVNldDtcbiAgfVxuXG4gIGFzeW5jIGRvd25sb2FkQXNKc29uKHBvc2VTZXQ6IFBvc2VTZXQpIHtcbiAgICBjb25zdCBibG9iID0gbmV3IEJsb2IoW2F3YWl0IHBvc2VTZXQuZ2V0SnNvbigpXSwge1xuICAgICAgdHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nLFxuICAgIH0pO1xuICAgIGNvbnN0IHVybCA9IHdpbmRvdy5VUkwuY3JlYXRlT2JqZWN0VVJMKGJsb2IpO1xuICAgIGNvbnN0IGEgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdhJyk7XG4gICAgYS5ocmVmID0gdXJsO1xuICAgIGEudGFyZ2V0ID0gJ19ibGFuayc7XG4gICAgYS5kb3dubG9hZCA9IGAke3Bvc2VTZXQuZ2V0VmlkZW9OYW1lKCl9LnBvc2VzLmpzb25gO1xuICAgIGEuY2xpY2soKTtcbiAgfVxuXG4gIGFzeW5jIGRvd25sb2FkQXNaaXAocG9zZVNldDogUG9zZVNldCkge1xuICAgIGNvbnN0IGNvbnRlbnQgPSBhd2FpdCBwb3NlU2V0LmdldFppcCgpO1xuICAgIGNvbnN0IHVybCA9IHdpbmRvdy5VUkwuY3JlYXRlT2JqZWN0VVJMKGNvbnRlbnQpO1xuICAgIGNvbnN0IGEgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdhJyk7XG4gICAgYS5ocmVmID0gdXJsO1xuICAgIGEudGFyZ2V0ID0gJ19ibGFuayc7XG4gICAgYS5kb3dubG9hZCA9IGAke3Bvc2VTZXQuZ2V0VmlkZW9OYW1lKCl9LnBvc2VzLnppcGA7XG4gICAgYS5jbGljaygpO1xuICB9XG59XG4iXX0=