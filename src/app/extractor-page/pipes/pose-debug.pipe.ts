import { Pipe, PipeTransform } from '@angular/core';
import { PoseSetItem } from 'ngx-mp-pose-extractor';

@Pipe({
  name: 'poseDebug',
})
export class PoseDebugPipe implements PipeTransform {
  transform(item: PoseSetItem, ...args: unknown[]): string {
    if (item.debug === undefined || item.debug.duplicatedItems === undefined) {
      return '';
    }

    let result = '[重複]\n';
    if (item.debug.duplicatedItems.length === 0) {
      return result + 'N/A';
    }
    for (const duplicatedItem of item.debug.duplicatedItems) {
      result += `・${duplicatedItem.timeMiliseconds}`;
      if (duplicatedItem.bodySimilarity !== undefined) {
        result += `　・bodySim: ${duplicatedItem.bodySimilarity}
　・handSim: ${duplicatedItem.handSimilarity ?? 'N/A'}`;
      }
    }

    return result;
  }
}
