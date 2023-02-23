import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MaterialModule } from 'src/material.module';
import { ExtractorPageComponent } from './extractor-page/extractor-page.component';
import { TesterPageComponent } from './tester-page/tester-page.component';
import { TesterComponent } from './tester-page/tester.component';
import { NgxMpPoseExtractorModule } from 'ngx-mp-pose-extractor';
import { FormsModule } from '@angular/forms';
import { ExtendedClassifierComponent } from './extractor-page/widgets/extended-classifier/extended-classifier.component';

@NgModule({
  declarations: [
    AppComponent,
    ExtractorPageComponent,
    TesterPageComponent,
    TesterComponent,
    ExtendedClassifierComponent,
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    BrowserAnimationsModule,
    MaterialModule,
    NgxMpPoseExtractorModule,
  ],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}
