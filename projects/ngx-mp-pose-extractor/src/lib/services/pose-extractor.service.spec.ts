import { TestBed } from '@angular/core/testing';

import { PoseExtractorService } from './pose-extractor.service';

describe('PoseExtractorService', () => {
  let service: PoseExtractorService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PoseExtractorService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
