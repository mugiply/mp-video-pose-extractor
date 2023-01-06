import { TestBed } from '@angular/core/testing';

import { PoseExtractorService } from './pose-extractor.service';

describe('ExtractorService', () => {
  let service: PoseExtractorService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PoseExtractorService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
